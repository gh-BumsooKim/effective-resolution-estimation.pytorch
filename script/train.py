"""Self-supervised pre-training of the effective-resolution estimator.

Usage
-----
    python script/train.py --resol 256 --dataset_limit 10000

Each training image is turned into one degraded sample (see
:mod:`dataset.image_dataset`), the background is masked out with BiSeNet, and
foreground patches are extracted.  For every patch we take a regular and an
adversarial (PGD) gradient step with equal weight, accumulating gradients over
``batch_accumulation`` (512) patches before an optimiser update.
"""

import os
import sys
import time
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config.params_config import training_default, training_config
from dataset.image_dataset import ImageLoader
from model.eff_resnet import EffResNet
from model.masking import FaceMasker
from model.pgd import PGD
from utils.patches import extract_patches
from utils.train_util import mape_loss, reset_LR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-r', '--resol', default='256', choices=['128', '256'])
    p.add_argument('-g', '--num_gpu', default='0')
    p.add_argument('--dataset_path', default=None,
                   help='override training_default["train_dataset_path"]')
    p.add_argument('--dataset_limit', type=int, default=None,
                   help='use only the first N images (10000 for the subset run)')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--real_batch', type=int, default=None,
                   help='patches per forward pass (memory only; default from config)')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--checkpoint_dir', default=None)
    p.add_argument('--resume', default=None, help='checkpoint to resume from')
    p.add_argument('--max_steps', type=int, default=None,
                   help='stop after N optimiser steps (smoke testing)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def build_config(args):
    config = dict(training_default)
    config.update(training_config[f'resol{args.resol}'])
    if args.dataset_path is not None:
        config['train_dataset_path'] = args.dataset_path
    if args.dataset_limit is not None:
        config['dataset_limit'] = args.dataset_limit
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    if args.real_batch is not None:
        config['real_batch_size'] = args.real_batch
    if args.checkpoint_dir is not None:
        config['checkpoint_dir'] = args.checkpoint_dir
    return config


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # TF32: use tensor cores for conv/matmul (Ampere+).  Keeps FP32 dynamic
    # range (8-bit exponent) and FP32 accumulation, so quality is essentially
    # unchanged while conv-bound training gets a speed-up.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = build_config(args)
    device = torch.device(f'cuda:{args.num_gpu}'
                          if torch.cuda.is_available() else 'cpu')
    print(f'[setup] TF32 enabled (matmul={torch.backends.cuda.matmul.allow_tf32}, '
          f'cudnn={torch.backends.cudnn.allow_tf32})')
    print(f'[setup] device={device} resol={args.resol} '
          f'patch={config["patch_size"]} real_batch={config["real_batch_size"]}')

    # --- data --------------------------------------------------------------
    dataset = ImageLoader(config)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0)
    print(f'[data] {len(dataset)} images')

    # --- model / masker / optimiser ---------------------------------------
    model = EffResNet(config['backbone'], config['pretrained'],
                      config['final_activation']).to(device)
    masker = FaceMasker(device, input_size=config['bisenet_input_size'])
    pgd = PGD(model, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 betas=config['optim_beta'], eps=config['optim_eps'])

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f'[resume] from {args.resume} at epoch {start_epoch}')

    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    patch_size = config['patch_size']
    stride = config['patch_stride']
    real_batch = config['real_batch_size']
    accum_target = config['batch_accumulation']
    max_patches = config['max_train_patches']
    min_fg = config['pre_min_foreground']

    global_step = 0
    for epoch in range(start_epoch, config['num_epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        run_reg = run_adv = 0.0
        run_cnt = 0
        t0 = time.time()

        for it, (img, y, r) in enumerate(loader):
            img = img.to(device, non_blocking=True)[0]     # (3, H, W)
            y_val = float(y.item())

            # --- mask + patches -------------------------------------------
            mask = masker.foreground_mask(img)             # (1, 1, H, W)
            img_masked = img * mask[0]                      # zero background
            patches, _ = extract_patches(
                img_masked, mask[0], patch_size, stride,
                random_offset=config['random_patch_offset'], min_foreground=min_fg)
            n = patches.shape[0]
            if n == 0:
                continue
            if max_patches is not None and n > max_patches:
                idx = torch.randperm(n, device=patches.device)[:max_patches]
                patches = patches[idx]
                n = max_patches
            targets = torch.full((n,), y_val, device=device)

            # --- regular + adversarial steps over micro-batches -----------
            for s in range(0, n, real_batch):
                pb = patches[s:s + real_batch]
                tb = targets[s:s + real_batch]
                nb = pb.shape[0]
                weight = nb / accum_target

                pred = model(pb)
                loss_reg = mape_loss(pred, tb)

                adv = pgd(pb, tb)
                pred_adv = model(adv)
                loss_adv = mape_loss(pred_adv, tb)

                loss = 0.5 * (loss_reg + loss_adv) * weight
                loss.backward()

                run_reg += loss_reg.item() * nb
                run_adv += loss_adv.item() * nb
                run_cnt += nb
                accum += nb

                if accum >= accum_target:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    global_step += 1

                    if global_step % config['log_every'] == 0:
                        lr = optimizer.param_groups[0]['lr']
                        print(f'[e{epoch} step {global_step}] '
                              f'reg={run_reg / run_cnt:.4f} '
                              f'adv={run_adv / run_cnt:.4f} '
                              f'lr={lr:.2e} img={it + 1}/{len(dataset)} '
                              f'({(time.time() - t0) / (it + 1):.2f}s/img)',
                              flush=True)
                        run_reg = run_adv = 0.0
                        run_cnt = 0

                    if args.max_steps and global_step >= args.max_steps:
                        print('[stop] reached max_steps', flush=True)
                        _save(config, model, optimizer, epoch, args.resol, tag='smoke')
                        return

        # flush a partial accumulation window
        if accum > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        reset_LR(optimizer, config['lr_decay'])
        print(f'[epoch {epoch}] done in {(time.time() - t0) / 60:.1f} min', flush=True)
        if config['save_every_epoch']:
            _save(config, model, optimizer, epoch, args.resol, tag=f'epoch{epoch}')
        _save(config, model, optimizer, epoch, args.resol, tag='latest')

    print('[done] training complete', flush=True)


def _save(config, model, optimizer, epoch, resol, tag):
    path = os.path.join(config['checkpoint_dir'],
                        f'eff_resnet_resol{resol}_{tag}.pth')
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'resol': resol}, path)
    print(f'[save] {path}', flush=True)


if __name__ == '__main__':
    main()

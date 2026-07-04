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
    p.add_argument('--whole', action='store_true',
                   help='use the whole-face config (whole{resol}): resize the '
                        'whole face to a fixed size instead of tiling patches. '
                        'Best when inference uses fixed-size crops.')
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
    key = f'whole{args.resol}' if args.whole else f'resol{args.resol}'
    config.update(training_config[key])
    config['config_key'] = key
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
    mode = (f'whole-face {config["whole_image_size"]}px'
            if config.get('whole_image_size') else 'patch (multi-scale)')
    print(f'[setup] device={device} config={config["config_key"]} mode={mode} '
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
    min_fg = (config['whole_min_foreground'] if config.get('whole_image_size')
              else config['pre_min_foreground'])

    def process_image(img):
        """Mask + extract (capped) patches for one image. Returns (P,3,P,P) or None."""
        mask = masker.foreground_mask(img)                 # (1, 1, H, W)
        img_masked = img * mask[0]                          # zero background
        p, _ = extract_patches(
            img_masked, mask[0], patch_size, stride,
            random_offset=config['random_patch_offset'], min_foreground=min_fg)
        if p.shape[0] == 0:
            return None
        if max_patches is not None and p.shape[0] > max_patches:
            sel = torch.randperm(p.shape[0], device=p.device)[:max_patches]
            p = p[sel]
        return p

    def optimizer_step(Pb, Tb):
        """One optimiser update over a mixed-image batch of len(Pb) patches.

        Crucially the batch mixes patches from many images (different targets),
        so BatchNorm cannot read the shared label off the batch statistics --
        the model is forced to learn per-patch sharpness features.
        """
        optimizer.zero_grad(set_to_none=True)
        total = Pb.shape[0]
        reg_sum = adv_sum = 0.0
        for s in range(0, total, real_batch):
            pb, tb = Pb[s:s + real_batch], Tb[s:s + real_batch]
            nb = pb.shape[0]
            w = nb / total
            loss_reg = mape_loss(model(pb), tb)
            adv = pgd(pb, tb)
            loss_adv = mape_loss(model(adv), tb)
            (0.5 * (loss_reg + loss_adv) * w).backward()
            reg_sum += loss_reg.item() * nb
            adv_sum += loss_adv.item() * nb
        optimizer.step()
        return reg_sum / total, adv_sum / total

    global_step = 0
    for epoch in range(start_epoch, config['num_epochs'] + 1):
        model.train()
        t0 = time.time()
        buf_p, buf_t, buf_n = [], [], 0
        run_reg = run_adv = 0.0
        run_k = 0

        for it, (img, y, r) in enumerate(loader):
            img = img.to(device, non_blocking=True)[0]     # (3, H, W)
            p = process_image(img)
            if p is None:
                continue
            buf_p.append(p)
            buf_t.append(torch.full((p.shape[0],), float(y.item()), device=device))
            buf_n += p.shape[0]

            # once enough patches are buffered, form a shuffled mixed-image batch
            while buf_n >= accum_target:
                P = torch.cat(buf_p)
                T = torch.cat(buf_t)
                perm = torch.randperm(P.shape[0], device=P.device)
                bidx, rest = perm[:accum_target], perm[accum_target:]
                r_reg, r_adv = optimizer_step(P[bidx], T[bidx])
                global_step += 1

                # carry the leftover patches over to the next batch
                buf_p = [P[rest]] if rest.numel() else []
                buf_t = [T[rest]] if rest.numel() else []
                buf_n = int(rest.numel())

                run_reg += r_reg
                run_adv += r_adv
                run_k += 1
                if global_step % config['log_every'] == 0:
                    lr = optimizer.param_groups[0]['lr']
                    print(f'[e{epoch} step {global_step}] '
                          f'reg={run_reg / run_k:.4f} adv={run_adv / run_k:.4f} '
                          f'lr={lr:.2e} img={it + 1}/{len(dataset)} '
                          f'({(time.time() - t0) / (it + 1):.2f}s/img)', flush=True)
                    run_reg = run_adv = 0.0
                    run_k = 0

                if args.max_steps and global_step >= args.max_steps:
                    print('[stop] reached max_steps', flush=True)
                    _save(config, model, optimizer, epoch, args.resol, tag='smoke')
                    return

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

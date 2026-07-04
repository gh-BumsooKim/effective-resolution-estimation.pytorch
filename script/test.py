"""Effective-resolution estimation (inference, Fig. 3b).

    python script/test.py --checkpoint checkpoints/eff_resnet_resol256_latest.pth \
                          --image path/to/face.png

The background is masked out, the image is split into ~100 patches, patches with
less than 90% foreground are dropped, each remaining patch is scored, and the
patch scores are aggregated with the median into a single ratio y.  The predicted
effective resolution is ``r_eff = y * r`` where ``r`` is the input resolution.
"""

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as T
from PIL import Image

from config.params_config import training_default, training_config
from model.eff_resnet import EffResNet
from model.masking import FaceMasker
from utils.patches import extract_patches, dynamic_stride


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-r', '--resol', default='256', choices=['128', '256'])
    p.add_argument('--whole', action='store_true',
                   help='use the whole-face config: resize each input to the '
                        'fixed size and score it as a single whole image '
                        '(matches training done with --whole).')
    p.add_argument('-g', '--num_gpu', default='0')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', default=None, help='single image')
    p.add_argument('--input_dir', default=None, help='directory of images')
    p.add_argument('--output_csv', default=None,
                   help='optional path to write results as CSV')
    p.add_argument('--mask', action='store_true',
                   help='enable BiSeNet background masking + foreground '
                        'filtering (OFF by default). Without it every patch is '
                        'scored, which suits tight face crops such as 256x256 '
                        'images (1 patch = whole image).')
    return p.parse_args()


@torch.no_grad()
def estimate(model, masker, config, device, path):
    pil = Image.open(path).convert('RGB')

    # whole-face mode: resize the whole image to the fixed size and score it as
    # one image (a single patch == the whole face), matching whole-face training.
    S = config.get('whole_image_size')
    if S is not None:
        pil = pil.resize((S, S), Image.BICUBIC)

    transform = T.Compose([
        #T.Resize((512, 512)),
        T.ToTensor()
    ])
    img = transform(pil).to(device)          # (3, H, W)
    r = img.shape[1]                            # height as absolute resolution

    P = config['patch_size']
    stride = dynamic_stride(img.shape[1], img.shape[2], P, config['num_patches'])

    if masker is not None:
        mask = masker.foreground_mask(img)[0]   # (1, H, W)
        img_masked = img * mask
        min_fg = config['post_min_foreground']
    else:
        # no masking: keep the whole image, accept every patch
        mask = torch.ones(1, img.shape[1], img.shape[2], device=device)
        img_masked = img
        min_fg = 0.0

    patches, _ = extract_patches(img_masked, mask, P, stride,
                                 random_offset=False,
                                 min_foreground=min_fg)
    if patches.shape[0] == 0:
        return None, 0

    scores = []
    for s in range(0, patches.shape[0], 64):
        scores.append(model(patches[s:s + 64], clip=True))
    scores = torch.cat(scores)

    if config['patch_score_aggregation'] == 'median':
        y = scores.median()
    else:
        y = scores.mean()
    return float(y) * r, patches.shape[0]


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = dict(training_default)
    config.update(training_config[f'whole{args.resol}' if args.whole
                                  else f'resol{args.resol}'])
    device = torch.device(f'cuda:{args.num_gpu}'
                          if torch.cuda.is_available() else 'cpu')

    model = EffResNet(config['backbone'], pretrained=False,
                      final_activation=config['final_activation']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()
    masker = FaceMasker(
        device, input_size=config['bisenet_input_size']) if args.mask else None

    if args.image:
        paths = [args.image]
    elif args.input_dir:
        paths = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    else:
        raise SystemExit('provide --image or --input_dir')

    rows = []
    for path in paths:
        r_eff, npatch = estimate(model, masker, config, device, path)
        if r_eff is None:
            print(f'{os.path.basename(path)}: no foreground patches')
        else:
            print(f'{os.path.basename(path)}: r_eff = {r_eff:.1f} '
                  f'(from {npatch} patches)')
        rows.append((path, '' if r_eff is None else f'{r_eff:.4f}', npatch))

    if args.output_csv:
        import csv
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'r_eff', 'num_patches'])
            writer.writerows(rows)
        print(f'[saved] {args.output_csv}')


if __name__ == '__main__':
    main()

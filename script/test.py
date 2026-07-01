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
    p.add_argument('-g', '--num_gpu', default='0')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', default=None, help='single image')
    p.add_argument('--input_dir', default=None, help='directory of images')
    return p.parse_args()


@torch.no_grad()
def estimate(model, masker, config, device, path):
    pil = Image.open(path).convert('RGB')
    img = T.ToTensor()(pil).to(device)          # (3, H, W)
    r = img.shape[1]                            # height as absolute resolution

    mask = masker.foreground_mask(img)
    img_masked = img * mask[0]

    P = config['patch_size']
    stride = dynamic_stride(img.shape[1], img.shape[2], P, config['num_patches'])
    patches, _ = extract_patches(img_masked, mask[0], P, stride,
                                 random_offset=False,
                                 min_foreground=config['post_min_foreground'])
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
    config.update(training_config[f'resol{args.resol}'])
    device = torch.device(f'cuda:{args.num_gpu}'
                          if torch.cuda.is_available() else 'cpu')

    model = EffResNet(config['backbone'], pretrained=False,
                      final_activation=config['final_activation']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()
    masker = FaceMasker(device, input_size=config['bisenet_input_size'])

    if args.image:
        paths = [args.image]
    elif args.input_dir:
        paths = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    else:
        raise SystemExit('provide --image or --input_dir')

    for path in paths:
        r_eff, npatch = estimate(model, masker, config, device, path)
        if r_eff is None:
            print(f'{os.path.basename(path)}: no foreground patches')
        else:
            print(f'{os.path.basename(path)}: r_eff = {r_eff:.1f} '
                  f'(from {npatch} patches)')


if __name__ == '__main__':
    main()

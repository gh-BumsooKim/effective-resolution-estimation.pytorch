"""Patch extraction with foreground filtering.

The background is masked out (set to 0) beforehand; here we cut the image into
``patch_size`` patches on a ``stride`` grid with a random start offset and keep
only patches whose foreground fraction is at least ``min_foreground``.
"""

import math
import random

import torch


def dynamic_stride(H, W, patch_size, target_num):
    """Stride giving roughly ``target_num`` patch positions over the image."""
    per_dim = max(1, int(round(math.sqrt(target_num))))
    stride_h = max(1, (H - patch_size) // per_dim) if H > patch_size else patch_size
    stride_w = max(1, (W - patch_size) // per_dim) if W > patch_size else patch_size
    return min(stride_h, stride_w)


def _positions(length, patch_size, stride, offset):
    if length < patch_size:
        return []
    start = offset % stride if stride > 0 else 0
    pos = list(range(start, length - patch_size + 1, stride))
    if not pos:
        pos = [0]
    # make sure the far edge is covered
    if pos[-1] != length - patch_size:
        pos.append(length - patch_size)
    return pos


def extract_patches(img, mask, patch_size, stride,
                    random_offset=True, min_foreground=0.5):
    """Cut ``img`` into foreground patches.

    ``img``  : (3, H, W)  in [0, 1] (background already zeroed).
    ``mask`` : (1, H, W)  foreground indicator in {0, 1}.

    Returns ``(patches, fractions)`` where ``patches`` is (N, 3, P, P) and
    ``fractions`` is (N,) -- both empty tensors if nothing qualifies.
    """
    if img.dim() == 4:
        img = img.squeeze(0)
    if mask.dim() == 4:
        mask = mask.squeeze(0)
    _, H, W = img.shape

    offx = random.randint(0, max(0, stride - 1)) if random_offset else 0
    offy = random.randint(0, max(0, stride - 1)) if random_offset else 0

    ys = _positions(H, patch_size, stride, offy)
    xs = _positions(W, patch_size, stride, offx)

    patches, fractions = [], []
    for y in ys:
        for x in xs:
            frac = mask[:, y:y + patch_size, x:x + patch_size].mean()
            if frac.item() >= min_foreground:
                patches.append(img[:, y:y + patch_size, x:x + patch_size])
                fractions.append(frac)

    if not patches:
        return (img.new_zeros((0, 3, patch_size, patch_size)),
                img.new_zeros((0,)))
    return torch.stack(patches, 0), torch.stack(fractions, 0)

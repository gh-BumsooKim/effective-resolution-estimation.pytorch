"""Self-supervised training-sample generation.

For every image we build one training sample following Sec. 4.1 of the paper:

1. (optional, 80%)  *prescale* the sharp image to a random resolution so the
   network becomes robust to different face sizes.  Prescaling above the native
   resolution can only add redundancy, so the base effective-resolution ratio is
   ``min(1, native / prescaled)``.
2. (optional, 90%)  *degrade* by down-scaling with a random factor and a random
   interpolation method, then up-scaling back to the original resolution with
   another random interpolation method.  The degradation ratio equals the
   down-scaling factor ``r_down / r_up``.
3. The regression target is the resulting effective-resolution ratio
   ``y = min(base_ratio, degrade_ratio)`` -- the lowest resolution the sample
   is informationally equivalent to, divided by its absolute resolution.

The masking / patch extraction happens on the GPU inside the training loop
(BiSeNet needs a GPU), so ``__getitem__`` returns the full degraded image.
"""

import glob
import math
import random

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from train_util.interpolation_methods import resize, INTERPOLATION_METHODS


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, config):
        paths = sorted(glob.glob(config['train_dataset_path']))
        if not paths:
            raise FileNotFoundError(
                f"no images matched {config['train_dataset_path']!r}")
        limit = config.get('dataset_limit')
        if limit is not None:
            paths = paths[:limit]
        self.img_paths = paths

        self.interpolation = config['interpolation_methods']
        self.max_down = config['max_downscale_factor']
        self.prescale_range = config['prescale_range']
        self.prescale_freq = config['prescale_frequency']
        self.downscale_freq = config['downscale_frequency']
        self.antialias = config['use_antialiasing']
        self.whole = config.get('whole_image_size')   # None -> patch mode

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def _rand_method(self):
        return random.choice(self.interpolation)

    def _whole_item(self, pil):
        """256-aligned whole-face sample: resize the whole face to ``self.whole``
        (a sharp reference at that resolution), then optionally degrade it by
        down- then up-scaling.  The target is the degradation ratio.
        """
        S = self.whole
        pil = resize(pil, (S, S), 'bicubic', antialias=True)   # sharp S-ref

        if random.random() < self.downscale_freq:
            factor = random.uniform(1.0 / self.max_down, 1.0)
            d = max(1, int(round(S * factor)))
            pil_down = resize(pil, (d, d), self._rand_method(),
                              antialias=self.antialias)
            pil = resize(pil_down, (S, S), self._rand_method(),
                         antialias=self.antialias)
            y = d / float(S)
        else:
            y = 1.0

        img = self.to_tensor(pil)                # (3, S, S) in [0, 1]
        return img, torch.tensor(y, dtype=torch.float32), S

    def __getitem__(self, idx):
        pil = Image.open(self.img_paths[idx])
        pil.load()
        pil = pil.convert('RGB')

        if self.whole is not None:
            return self._whole_item(pil)

        w0, h0 = pil.size
        native = h0                      # reference (height) resolution

        # --- 1) prescale -------------------------------------------------
        if random.random() < self.prescale_freq:
            s = random.randint(int(self.prescale_range[0]),
                               int(self.prescale_range[1]))
            new_w = max(1, int(round(w0 * s / h0)))
            pil = resize(pil, (new_w, s), 'bicubic', antialias=True)
            r = s
            base_ratio = min(1.0, native / float(s))
        else:
            r = h0
            base_ratio = 1.0

        # --- 2) degrade (down- then up-scale) ----------------------------
        cur_w, cur_h = pil.size
        if random.random() < self.downscale_freq:
            factor = random.uniform(1.0 / self.max_down, 1.0)
            down_w = max(1, int(round(cur_w * factor)))
            down_h = max(1, int(round(cur_h * factor)))

            pil_down = resize(pil, (down_w, down_h), self._rand_method(),
                              antialias=self.antialias)
            pil = resize(pil_down, (cur_w, cur_h), self._rand_method(),
                         antialias=self.antialias)
            degrade_ratio = down_h / float(cur_h)
        else:
            degrade_ratio = 1.0

        # --- 3) target ---------------------------------------------------
        y = min(base_ratio, degrade_ratio)

        img = self.to_tensor(pil)        # (3, H, W) float in [0, 1]
        return img, torch.tensor(y, dtype=torch.float32), r

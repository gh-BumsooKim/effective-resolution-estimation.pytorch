"""Hyper-parameters for the self-supervised effective-resolution estimator.

Values follow Kansy et al., "Self-Supervised Effective Resolution Estimation
with Adversarial Augmentations" (WACVW 2023), Table 1 of the supplementary
material.  Where a hyper-parameter has two values in the paper they are given
as ``patch256`` | ``patch128`` (left of the ``|`` is patch size 256, right is
patch size 128).
"""

import os

from train_util.interpolation_methods import INTERPOLATION_METHODS

# ---------------------------------------------------------------------------
# Settings that are shared by every patch-size configuration.
# ---------------------------------------------------------------------------
training_default = {
    # --- data -------------------------------------------------------------
    # Glob pattern for the training images.  Set it (in order of precedence) via
    # the ``--dataset_path`` CLI flag, the ``SSERE_DATA_GLOB`` environment
    # variable, or by editing the relative default below -- no machine-specific
    # absolute path is baked into the repo.
    'train_dataset_path': os.environ.get(
        'SSERE_DATA_GLOB',
        os.path.join('data', 'ffhq', 'images1024x1024', '*.png')),
    'dataset_limit'     : None,      # e.g. 10000 to use only the first N images

    # --- model ------------------------------------------------------------
    'backbone'          : 'resnet50',
    'pretrained'        : True,      # ImageNet pre-training (Table 1)
    'final_activation'  : None,      # None / linear -> identity

    # --- optimisation -----------------------------------------------------
    'num_epochs'        : 10,
    'batch_accumulation': 512,       # simulated batch size (patches)
    'loss_function'     : 'mape',    # mean absolute percentage error
    'optimizer'         : 'adam',
    'optim_beta'        : (0.9, 0.999),
    'optim_eps'         : 1e-8,
    'learning_rate'     : 1e-3,
    'lr_decay'          : 0.9,       # staircase decay factor, once per epoch

    # --- adversarial augmentation (PGD) -----------------------------------
    'ratio_regular_vs_adversarial_steps' : 1,   # equal frequency
    'ratio_weight_regular_vs_adversarial': 1,   # equal loss weight
    'adversarial_method'  : 'pgd',
    'number_of_pgd_steps' : 10,
    'pgd_norm'            : 'l2',    # step is taken in the L2 norm
    'epsilon_ball'        : 10,      # L_inf ball (in [0, 255]) the noise is projected to
    'random_init'         : False,  # no random initialisation inside the ball
    'clip_range'          : (0, 255),   # perturbed images are clipped to this range

    # --- pre-processing ---------------------------------------------------
    'patch_stride'          : 128,
    'random_patch_offset'   : True,
    'pre_background_masking' : True,
    'mask_model'            : 'bisenet',
    'pre_min_foreground'    : 0.50,     # >=50% foreground to keep a patch (train)
    'max_downscale_factor'  : 16,       # df_m
    'use_antialiasing'      : True,
    'interpolation_methods' : INTERPOLATION_METHODS,   # single source of truth
    'interpolation_sampling': 'uniform',
    'prescale_frequency'    : 0.80,     # 80% of samples are prescaled
    'downscale_frequency'   : 0.90,     # 90% of samples are downscaled (10% none)

    # Pragmatic addition (not in the paper): cap the number of patches drawn
    # from a single image per step so that one very large prescaled image does
    # not dominate an accumulation window.  Set to None to keep every patch.
    'max_train_patches'     : 32,

    # --- post-processing (inference) --------------------------------------
    'num_patches'           : 100,      # target number of patches at inference
    'post_background_masking': True,
    'post_min_foreground'   : 0.90,     # >=90% foreground to keep a patch (inference)
    'patch_score_aggregation': 'median',

    # --- misc -------------------------------------------------------------
    'bisenet_input_size'    : 512,      # resolution BiSeNet segments at
    'checkpoint_dir'        : 'checkpoints',
    'log_every'             : 20,       # log every N optimiser steps
    'save_every_epoch'      : True,
}

# ---------------------------------------------------------------------------
# Per patch-size overrides.
# ---------------------------------------------------------------------------
training_config = {
    # NOTE: The paper uses real_batch_size 4|16 (a Titan-X memory limit) and
    # accumulates gradients to a simulated batch of 512.  On a 24 GB card we use
    # a larger micro-batch -- this only changes speed/memory, not the
    # optimisation, since gradients are still accumulated to 512 patches.
    'resol256': {
        'patch_size'        : 256,
        'real_batch_size'   : 32,       # patches processed per forward pass
        'pgd_step_size'     : 30,       # L2 step size (in [0, 255])
        'prescale_range'    : (384, 2048),
    },
    'resol128': {
        'patch_size'        : 128,
        'real_batch_size'   : 64,
        'pgd_step_size'     : 15,
        'prescale_range'    : (256, 2048),
    },
}

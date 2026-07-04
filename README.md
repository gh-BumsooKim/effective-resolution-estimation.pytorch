# SS-ERE — Self-Supervised Effective Resolution Estimation (PyTorch)

Unofficial PyTorch implementation of

> M. Kansy, J. Balletshofer, J. Naruniec, C. Schroers, G. Mignone, M. Gross,
> R. M. Weber, **"Self-Supervised Effective Resolution Estimation with
> Adversarial Augmentations"**, WACVW 2023 (DisneyResearch|Studios).

The network learns to predict the *effective resolution ratio* `y = r_eff / r`
of a face image — the fraction of its absolute resolution that actually carries
detail (sharpness) — **without any human labels**. Training samples are
generated self-supervised by down- then up-scaling sharp images, and adversarial
(PGD) augmentations bridge the gap between synthetic and real degradations.

## Method (as implemented)

Training (`script/train.py`), per image:

1. **Prescale** (80% of samples) to a random resolution to be robust to face
   size; the target is adjusted since prescaling above the native resolution
   cannot add detail (`base_ratio = min(1, native / prescaled)`).
2. **Degrade** (90% of samples): downscale by a random factor
   `~ U(1/16, 1)` with a random interpolation method, then upscale back with
   another random method (antialiased). The regression target is
   `y = min(base_ratio, r_down / r_up)`.
3. **Mask** the background with BiSeNet (drop background/neck/necklace/cloth/hat,
   keep face + hair).
4. **Patches** of 256² (or 128²), stride 128, random offset; keep patches with
   ≥ 50% foreground. Each patch inherits the image's target `y`.
5. **Loss** = MAPE between per-patch predictions and `y`. Every micro-batch takes
   an equally-weighted **regular** and **adversarial** (PGD, 10 steps, L2 step,
   projected to an L∞ ball of 10/255) step. Gradients accumulate to a simulated
   batch of 512 patches before an Adam update (`lr 1e-3`, ×0.9 / epoch).

Inference (`script/test.py`): mask → ~100 patches (≥ 90% foreground) → per-patch
scores clipped to `[0, 1]` → **median** → `r_eff = y · r`.

## Setup

Tested with **Python 3.10** and **CUDA 12.1** on an RTX 3090 (24 GB).
Environment managed with [uv](https://docs.astral.sh/uv/).

```bash
# install uv first if needed:
#   curl -LsSf https://astral.sh/uv/install.sh | sh          (Linux/macOS)
#   powershell -c "irm https://astral.sh/uv/install.ps1|iex" (Windows)

# 1) create a virtual environment with Python 3.10
uv venv --python 3.10

# 2) activate it
#    Linux/macOS:  source .venv/bin/activate
#    Windows:      .venv\Scripts\activate

# 3) install dependencies (requirements.txt pulls the cu121 build of torch)
uv pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `numpy`, `opencv-python`, `pillow`
(exact tested versions are pinned in `requirements.txt`).

Before training you also need:
- the **BiSeNet** face-parsing weight at `face_parsing/res/cp/79999_iter.pth`
  (git-ignored — see [Pretrained weights](#pretrained-weights-not-committed));
- your face images, pointed to via `--dataset_path` / `SSERE_DATA_GLOB`.

The ResNet50 ImageNet weights download automatically on first run.

## Usage

Training (paper: 10 epochs; a 10 000-image FFHQ subset matches the paper's
"Subset of FFHQ" ablation):

Point it at your images with `--dataset_path` (or the `SSERE_DATA_GLOB`
environment variable):

```cmd
python script/train.py --resol 256 --dataset_path "/path/to/FFHQ/images1024x1024/*.png" ^
                       --dataset_limit 10000 --num_workers 8 --checkpoint_dir checkpoints
```

Resume: `--resume checkpoints/eff_resnet_resol256_latest.pth`.
Patch-size 128 variant: `--resol 128`.

Estimation:

```cmd
python script/test.py --resol 256 --checkpoint checkpoints/eff_resnet_resol256_latest.pth ^
                      --image path/to/face.png            # or --input_dir folder
```

Add `--no_mask` to skip BiSeNet background masking (and foreground filtering) and
score every patch — convenient for tight face crops such as 256×256 images, where
the whole image is a single patch. `r_eff = y * r` still scales with the input
resolution `r`, so keep the evaluation resolution consistent across images.

## Layout

| path | purpose |
|------|---------|
| `config/params_config.py` | all hyper-parameters (Table 1 of the supplement) |
| `dataset/image_dataset.py` | prescale + degrade + target generation |
| `train_util/interpolation_methods.py` | the 8 interpolation kernels (PIL + cv2) |
| `model/eff_resnet.py` | ResNet50 + FC(1), ImageNet-normalising wrapper |
| `model/masking.py` | BiSeNet foreground masking |
| `model/pgd.py` | PGD adversarial augmentation |
| `utils/patches.py` | patch extraction + foreground filtering |
| `utils/train_util.py` | MAPE loss, staircase LR decay |
| `script/train.py`, `script/test.py` | training / inference entry points |

## Notes on fidelity

- **Micro-batch size** is raised from the paper's 4|16 (a Titan-X memory limit)
  to 32|64 on a 24 GB card. This changes only speed/memory — gradients are still
  accumulated to a simulated batch of **512** patches, so the optimisation is
  unchanged.
- **Interpolation kernels** without an exact Pillow/OpenCV counterpart are
  approximated (`gaussian` → Gaussian pre-filter + bilinear, `lanczos5` →
  `INTER_LANCZOS4`, `mitchellcubic` → bicubic). The diversity of kernels plus the
  adversarial augmentation is what matters (Sec. 4.2), so this is acceptable.
- `max_train_patches` caps patches drawn per image so one large prescaled image
  cannot dominate an accumulation window (not in the paper; set `None` to disable).
- **TF32** is enabled (`torch.backends.*.allow_tf32 = True`) for a conv-bound
  speed-up on Ampere+ GPUs; it keeps FP32 range/accumulation so quality is
  unchanged. Comment it out in `script/train.py` for bit-exact FP32.

## Performance notes & roadmap

**Measured on the reference setup** (RTX 3090, Windows 10, PyTorch 2.5.1+cu121,
patch 256, micro-batch 32): ≈ 1.3 s/image, ~4 h/epoch (~40 h for the 10k×10 run).

- **TF32** (enabled) gives no measurable speed-up here — the workload is not
  FP32-matmul bound (more likely kernel-launch / memory-bandwidth bound). Kept on
  anyway since there is no downside.
- **`torch.compile`** is currently unavailable on this Windows environment:
  Inductor needs Triton, which has no working Windows install here. It works out
  of the box on Linux / WSL2.

Optimisation ideas (**not applied** — the default run stays faithful FP32):

| Idea | Expected gain | Fidelity impact | Notes |
|------|---------------|-----------------|-------|
| Patch batching across images | High (GPU util) | None | Biggest structural win; needs loop refactor |
| AMP bf16 | 1.3–1.7× | Slight (bf16 ≠ fp32) | Stable (no GradScaler); may help where TF32 didn't |
| `channels_last` (NHWC) | Small alone, more with AMP | None (identical) | One-liner |
| CUDA graphs for the PGD loop | Medium (launch overhead) | None | 10 identical steps fit well; static-shape constraint |
| GPU data aug (kornia / DALI) | Medium if CPU-bound | Depends on kernels | prescale/degrade up to 2048² is CPU-heavy |
| FGSM (1 step) vs 10-step PGD | High | Moderate | Paper: "slightly worse but speeds up tremendously" |
| `torch.compile` (Inductor) | 1.3–2× | None | Needs Triton (Linux / WSL2) |
| Custom CUDA / JAX rewrite | Low ROI here | — | cuDNN already optimal; JAX mainly shines on TPU |

**Fidelity roadmap** (paper features beyond the current implementation's scope):

- Validation-based **best-model selection** (the paper picks the best epoch on a
  validation set; here we just checkpoint every epoch).
- **Evaluation metrics** (SRCC / PRA) — require human-labelled test data, which is
  not publicly available.
- **Baselines** (frequency, compression) for comparison.
- A trained **patch-128** variant (config is present; not yet run).

## Pretrained weights (not committed)

`.pth` files are git-ignored, so after cloning you must obtain the BiSeNet
face-parsing weight used for masking:

- Download `79999_iter.pth` from the upstream
  [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
  repo and place it at `face_parsing/res/cp/79999_iter.pth`.

Effective-resolution checkpoints are produced by running the training above.

## Credits

- Method: Kansy et al., WACVW 2023 (see the official CVF open-access page for the
  paper and supplement — not redistributed here).
- `face_parsing/` is vendored from
  [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
  (MIT, see `face_parsing/LICENSE`).

This is an **unofficial** reimplementation and is not affiliated with the authors
or DisneyResearch|Studios.

## Acknowledgment

Implemented with [Claude Code](https://claude.com/claude-code) (Anthropic Claude).

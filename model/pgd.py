"""Projected gradient descent for the adversarial augmentations (Sec. 4.2).

10 PGD steps taking an ``L2``-normalised step of size 30 | 15 (in ``[0, 255]``
pixel units), projecting the accumulated noise onto an ``L_inf`` ball of radius
10 and clipping the perturbed image to ``[0, 255]``.  No random initialisation.

The gradient *ascends* the (MAPE) loss so the noise makes the sample look like a
different sharpness while its label stays fixed -- training on these examples
makes the model robust to the artificial-vs-real degradation domain gap.

All tensors here live in ``[0, 1]``; the step size / epsilon / clip range are
scaled by ``1/255`` accordingly.
"""

import torch

from utils.train_util import mape_loss


class PGD:
    def __init__(self, model, config):
        self.model = model
        self.steps = config['number_of_pgd_steps']
        self.step_size = config['pgd_step_size'] / 255.0    # L2 step in [0, 1]
        self.eps = config['epsilon_ball'] / 255.0           # L_inf ball in [0, 1]
        lo, hi = config['clip_range']
        self.clip_lo = lo / 255.0
        self.clip_hi = hi / 255.0
        self.random_init = config['random_init']

    def __call__(self, x, y):
        """``x``: (N, 3, P, P) in [0, 1].  ``y``: (N,) targets.  Returns x_adv."""
        was_training = self.model.training
        self.model.eval()               # freeze BN stats while generating noise

        x_orig = x.detach()
        x_adv = x_orig.clone()
        if self.random_init:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = (x_orig + (x_adv - x_orig).clamp(-self.eps, self.eps))
            x_adv = x_adv.clamp(self.clip_lo, self.clip_hi)

        n = x_orig.shape[0]
        for _ in range(self.steps):
            x_adv = x_adv.detach().requires_grad_(True)
            pred = self.model(x_adv)
            loss = mape_loss(pred, y)                       # to be maximised
            grad = torch.autograd.grad(loss, x_adv)[0]

            # normalise the gradient in the L2 norm (per sample)
            flat = grad.view(n, -1)
            norm = flat.norm(dim=1).clamp_min(1e-12).view(n, 1, 1, 1)
            x_adv = x_adv.detach() + self.step_size * grad / norm

            # project the noise onto the L_inf ball and clip to a valid image
            delta = (x_adv - x_orig).clamp(-self.eps, self.eps)
            x_adv = (x_orig + delta).clamp(self.clip_lo, self.clip_hi)

        if was_training:
            self.model.train()
        return x_adv.detach()

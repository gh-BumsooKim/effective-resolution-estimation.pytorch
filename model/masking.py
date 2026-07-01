"""Foreground (face) masking with a modified BiSeNet.

The paper masks out the background before extracting patches, because the
downscaling-implies-sharpness assumption only holds for the face and because
generated images have complex background degradations that should be ignored.

The following BiSeNet (CelebAMask-HQ, 19 classes) labels are considered
*background* and masked out (Sec. 1 of the supplementary material):

    0 background, 14 neck, 15 neck_l (necklace), 16 cloth, 18 hat

Everything else -- skin, brows, eyes, glasses, ears, earring, nose, mouth,
lips and **hair** -- is kept as foreground.
"""

import os
import importlib.util
import sys

import torch
import torch.nn.functional as F


# BiSeNet labels kept as foreground (face region incl. hair).
FOREGROUND_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17]

_FACE_PARSING_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'face_parsing')
_DEFAULT_CKPT = os.path.join(_FACE_PARSING_DIR, 'res', 'cp', '79999_iter.pth')

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_bisenet_class():
    """Import ``BiSeNet`` from ``face_parsing/model.py``.

    ``face_parsing/model.py`` does ``from resnet import Resnet18`` so the
    directory must be importable, and :class:`Resnet18` downloads ImageNet
    weights on construction -- redundant here (we load the full checkpoint),
    so we stub out the download.
    """
    # Append (not prepend) so face_parsing/model.py cannot shadow the repo's
    # own ``model`` package; ``from resnet import Resnet18`` still resolves.
    if _FACE_PARSING_DIR not in sys.path:
        sys.path.append(_FACE_PARSING_DIR)

    import torch.utils.model_zoo as model_zoo
    model_zoo.load_url = lambda *a, **k: {}   # avoid the redundant download

    spec = importlib.util.spec_from_file_location(
        'face_parsing_model', os.path.join(_FACE_PARSING_DIR, 'model.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BiSeNet


class FaceMasker:
    def __init__(self, device, checkpoint=_DEFAULT_CKPT, input_size=512,
                 foreground_labels=FOREGROUND_LABELS):
        self.device = device
        self.input_size = input_size
        self.foreground_labels = torch.tensor(foreground_labels, device=device)

        BiSeNet = _load_bisenet_class()
        net = BiSeNet(n_classes=19)
        state = torch.load(checkpoint, map_location='cpu')
        net.load_state_dict(state)
        net.eval().to(device)
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net

        self.mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def foreground_mask(self, img):
        """``img``: (3, H, W) or (1, 3, H, W) in [0, 1] on ``device``.

        Returns a (1, 1, H, W) float mask with 1 = foreground, 0 = background.
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)
        _, _, H, W = img.shape

        seg_in = F.interpolate(img, size=(self.input_size, self.input_size),
                               mode='bilinear', align_corners=False)
        seg_in = (seg_in - self.mean) / self.std
        logits = self.net(seg_in)[0]                       # (1, 19, S, S)
        parsing = logits.argmax(dim=1, keepdim=True).float()   # (1, 1, S, S)

        mask = torch.isin(parsing, self.foreground_labels.float()).float()
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        return mask

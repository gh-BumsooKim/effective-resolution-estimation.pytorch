"""Non-parametric down-/up-scaling used to build training samples.

The paper samples the interpolation method uniformly from::

    area, bicubic, bilinear, gaussian, lanczos3, lanczos5, mitchellcubic,
    nearest neighbour

The reference implementation relied on ``tf.keras.layers.Resizing`` which
offers all of those kernels.  We reproduce them with Pillow (already used by
the original repo) and OpenCV.  Three kernels have no exact Pillow/OpenCV
counterpart and are approximated as documented below -- this is acceptable
because the *diversity* of interpolation methods is what matters for the
adversarial augmentation, and the adversarial noise is what bridges the
remaining domain gap (see Sec. 4.2 of the paper).

    gaussian       -> Gaussian pre-filter (antialias) + bilinear resize
    lanczos5       -> OpenCV INTER_LANCZOS4  (a=4, closest available)
    mitchellcubic  -> Pillow BICUBIC         (both are cubic filters)
    area           -> OpenCV INTER_AREA
    lanczos3       -> Pillow LANCZOS         (a=3, exact)
"""

import numpy as np
from PIL import Image
import cv2

# Interpolation methods that the paper samples from.
INTERPOLATION_METHODS = [
    'area', 'bicubic', 'bilinear', 'gaussian',
    'lanczos3', 'lanczos5', 'mitchellcubic', 'nearest',
]

_PIL_FILTERS = {
    'bicubic':       Image.BICUBIC,
    'bilinear':      Image.BILINEAR,
    'nearest':       Image.NEAREST,
    'lanczos3':      Image.LANCZOS,      # Pillow Lanczos uses a = 3
    'mitchellcubic': Image.BICUBIC,      # approximation (see module docstring)
}

_CV2_FILTERS = {
    'area':     cv2.INTER_AREA,
    'lanczos5': cv2.INTER_LANCZOS4,      # approximation (a = 4)
}


def _gaussian_resize(pil_img, size, antialias):
    """Gaussian-kernel resize (TF's ``gaussian`` interpolation, approximated).

    ``size`` is ``(width, height)``.
    """
    src_w, src_h = pil_img.size
    dst_w, dst_h = size
    arr = np.asarray(pil_img)

    downscaling = dst_w < src_w or dst_h < src_h
    if downscaling and antialias:
        # Anti-alias by low-pass filtering before decimation.  sigma is chosen
        # proportional to the (largest) downscaling factor.
        factor = max(src_w / max(dst_w, 1), src_h / max(dst_h, 1))
        sigma = max(1e-3, 0.5 * (factor - 1.0))
        arr = cv2.GaussianBlur(arr, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)

    out = cv2.resize(arr, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(out)


def resize(pil_img, size, method, antialias=True):
    """Resize ``pil_img`` to ``size`` (``(width, height)``) with ``method``.

    Always returns a :class:`PIL.Image.Image`.  ``antialias`` only affects
    down-scaling; Pillow's high-quality filters and OpenCV ``INTER_AREA``
    already low-pass filter when reducing.
    """
    size = (int(round(size[0])), int(round(size[1])))
    if size[0] < 1 or size[1] < 1:
        raise ValueError(f'invalid target size {size}')

    if method in _PIL_FILTERS:
        return pil_img.resize(size, resample=_PIL_FILTERS[method])

    if method in _CV2_FILTERS:
        arr = np.asarray(pil_img)
        out = cv2.resize(arr, (size[0], size[1]), interpolation=_CV2_FILTERS[method])
        return Image.fromarray(out)

    if method == 'gaussian':
        return _gaussian_resize(pil_img, size, antialias)

    raise ValueError(f'unsupported interpolation method: {method!r}')

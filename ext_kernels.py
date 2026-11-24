"""Additional OOD kernel families for the shift study (evaluation only, no retraining).

defocus : uniform disk (out-of-focus) kernels, antialiased by 8x supersampling, radii 1.5-9 px.
levin   : the 8 real camera-shake kernels of Levin et al. (CVPR 2009), measured from real captures.
          Models support 21x21 kernels. Each kernel is centred on its (rounded) centroid and cropped to 21x21.
          Kernel 4 (27x27) keeps only 89 % of its mass inside 21x21 and is EXCLUDED; kernels 7-8
          (23x23) keep >= 99.8 % and are cropped then renormalised.
          The .mat files store the kernel for correlation; it is rotated by 180 degrees here so that
          y ~ k * x under true convolution (checked: residual std 1.3-1.9 grey levels vs 3-9 unrotated).
Data: data/levin09/Levin09blurdata/imXX_flitYY.mat (x sharp, y real blurred, f kernel).
"""

from paths import LEVIN_DIR  # noqa: E402
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import center_of_mass

from degrade_torch import KSIZE

LEVIN_KEEP = (1, 2, 3, 5, 6, 7, 8)          # kernel 4 excluded (exceeds 21x21 support)
DEFOCUS_RADII = (1.5, 2.5, 3.5, 5.0, 7.0, 9.0)


def disk_kernel(radius, size=KSIZE, ss=8):
    c = (size - 1) / 2
    g = (np.arange(size * ss) + 0.5) / ss - 0.5 - c
    yy, xx = np.meshgrid(g, g, indexing="ij")
    k = (xx ** 2 + yy ** 2 <= radius ** 2).astype(np.float64)
    k = k.reshape(size, ss, size, ss).mean((1, 3))
    return (k / k.sum()).astype(np.float32)


def fit_kernel(k, size=KSIZE):
    """Centre a kernel on its centroid and crop/pad it to size x size; returns (kernel, kept mass)."""
    pad = 2 * size + 1
    kp = np.zeros((pad, pad))
    n = k.shape[0]
    o = (pad - n) // 2
    kp[o:o + n, o:o + n] = k / k.sum()
    cy, cx = center_of_mass(kp)
    # integer shift only: sub-pixel interpolation would blur the measured kernel
    kp = np.roll(kp, (int(round(pad // 2 - cy)), int(round(pad // 2 - cx))), (0, 1))
    c, h = pad // 2, size // 2
    out = kp[c - h:c + h + 1, c - h:c + h + 1]
    kept = out.sum() / kp.sum()
    return (out / out.sum()).astype(np.float32), float(kept)


def levin_kernels():
    return np.stack([fit_kernel(loadmat(LEVIN_DIR / f"im05_flit{i:02d}.mat")["f"][::-1, ::-1])[0] for i in LEVIN_KEEP])


def shift_kernel_sets():
    return {"defocus": np.stack([disk_kernel(r) for r in DEFOCUS_RADII]), "levin": levin_kernels()}


def levin_real_pairs():
    """Real captured pairs: (name, sharp x, real blurred y, fitted kernel), kernel 4 excluded."""
    out = []
    for im in (5, 6, 7, 8):
        for i in LEVIN_KEEP:
            d = loadmat(LEVIN_DIR / f"im{im:02d}_flit{i:02d}.mat")
            out.append((f"im{im:02d}_k{i}", d["x"].astype(np.float32), d["y"].astype(np.float32),
                        fit_kernel(d["f"][::-1, ::-1])[0]))
    return out

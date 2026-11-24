"""Batched GPU degradations for the two formation orders, plus fixed kernel sets.

Model (all values on the 0-1 scale, noise sigmas given in 0-255 units):
  blur->noise : y = k * x + sigma * n1                + floor * n2
  noise->blur : y = k * x + sigma_pre * (k * n1)      + floor * n2
sigma_pre = sigma (same_input_sigma) or sigma / ||k||_2 (matched_effective_energy).
The two orders share n1 and n2, so paired observations differ only in the order.
Convolution uses reflect padding (not circular), unlike the Wiener pilot.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

MODES = ("same_input_sigma", "matched_effective_energy")
KSIZE = 21


def gaussian_kernel(sx, sy=None, theta=0.0, size=KSIZE):
    """Anisotropic Gaussian; widths <= 0 give a delta kernel."""
    sy = sx if sy is None else sy
    k = np.zeros((size, size))
    if sx <= 0 or sy <= 0:
        k[size // 2, size // 2] = 1
        return k
    r = np.arange(size) - size // 2
    xx, yy = np.meshgrid(r, r)
    c, s = math.cos(theta), math.sin(theta)
    u, v = c * xx + s * yy, -s * xx + c * yy
    k = np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    return k / k.sum()


def random_gaussian_kernel(rng, width_range=(0.2, 3.0), p_iso=0.5):
    lo, hi = width_range
    if rng.random() < p_iso:
        w = rng.uniform(lo, hi)
        return gaussian_kernel(w)
    return gaussian_kernel(rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(0, math.pi))


def motion_kernel(rng, size=KSIZE, length_range=(5, 15)):
    """Random camera-shake trajectory rasterized with bilinear splatting."""
    length = rng.uniform(*length_range)
    steps = 200
    angle = rng.uniform(0, 2 * math.pi)
    v = np.array([math.cos(angle), math.sin(angle)])
    p = np.zeros(2)
    traj = [p.copy()]
    for _ in range(steps):
        turn = rng.normal(0, 0.25)
        c, s = math.cos(turn), math.sin(turn)
        v = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        p = p + v + rng.normal(0, 0.15, 2)
        traj.append(p.copy())
    traj = np.array(traj)
    traj -= traj.mean(0)
    span = np.linalg.norm(traj.max(0) - traj.min(0))
    traj *= length / max(span, 1e-6)
    traj += size // 2
    k = np.zeros((size, size))
    for x, y in traj:
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                          (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= x0 + dx < size and 0 <= y0 + dy < size:
                k[y0 + dy, x0 + dx] += w
    return k / k.sum()


def eval_kernel_sets(motion_seed=12345, n_motion=8):
    """Locked evaluation kernels. 'gauss' is in-distribution, 'motion' is OOD."""
    gauss = [gaussian_kernel(w) for w in (0.5, 1.0, 2.0, 3.0)]
    gauss += [gaussian_kernel(0.6, 2.0, math.pi / 6), gaussian_kernel(1.0, 3.0, 2 * math.pi / 3)]
    rng = np.random.default_rng(motion_seed)
    motion = [motion_kernel(rng) for _ in range(n_motion)]
    return {"gauss": np.stack(gauss), "motion": np.stack(motion)}


def _blur(x, kernels):
    """x: (B,C,H,W); kernels: (B,K,K). Per-sample kernel, reflect padding."""
    b, c, h, w = x.shape
    ks = kernels.shape[-1]
    weight = kernels.to(x.dtype).repeat_interleave(c, dim=0)[:, None]
    xp = F.pad(x.reshape(1, b * c, h, w), [ks // 2] * 4, mode="reflect")
    # conv2d is cross-correlation; flip so this is a true convolution.
    return F.conv2d(xp, weight.flip(-1, -2), groups=b * c).reshape(b, c, h, w)


def degrade_pair(x, kernels, sigma, mode, floor=0.0, generator=None):
    """Return (blur->noise, noise->blur) observations that share all noise draws.

    sigma: (B,) tensor in 0-255 units. floor: scalar in 0-255 units.
    """
    if mode not in MODES:
        raise ValueError(mode)
    n1 = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    n2 = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    s = (sigma / 255).to(x.dtype)[:, None, None, None]
    s_pre = s
    if mode == MODES[1]:
        knorm = kernels.flatten(1).norm(dim=1).to(x.dtype)[:, None, None, None]
        s_pre = s / knorm
    kx = _blur(x, kernels)
    floor_noise = (floor / 255) * n2
    bn = kx + s * n1 + floor_noise
    nb = kx + s_pre * _blur(n1, kernels) + floor_noise
    return bn, nb


N_DESC = 5


def degradation_descriptor(kernels, sigma, present=None):
    """Continuous descriptor defined for any kernel: [present, sigma/40, sx/3, sy/3, sxy/9].

    sx, sy, sxy come from the kernel's second moments, so motion kernels (OOD) get a
    descriptor in the same space as the Gaussian training kernels.
    kernels: (B,K,K) tensor, sigma: (B,) in 0-255 units, present: (B,) bool or None.
    """
    b, ks, _ = kernels.shape
    r = torch.arange(ks, device=kernels.device, dtype=kernels.dtype) - ks // 2
    k = kernels / kernels.sum((1, 2), keepdim=True)
    mx = (k.sum(1) * r).sum(1)
    my = (k.sum(2) * r).sum(1)
    sxx = (k.sum(1) * r ** 2).sum(1) - mx ** 2
    syy = (k.sum(2) * r ** 2).sum(1) - my ** 2
    sxy = (k * r[:, None] * r[None, :]).sum((1, 2)) - mx * my
    desc = torch.stack([torch.ones_like(sigma), sigma / 40, sxx.clamp_min(0).sqrt() / 3,
                        syy.clamp_min(0).sqrt() / 3, sxy / 9], 1)
    if present is not None:
        desc = desc * present.to(desc.dtype)[:, None]
    return desc


N_KDESC = 2 + KSIZE * KSIZE


def kernel_descriptor(kernels, sigma, present=None):
    """Lossless descriptor: [present, sigma/40, sqrt(k) flattened].

    Control for degradation_descriptor: the full kernel is given, so an OOD (motion)
    kernel is described exactly rather than squeezed into Gaussian moments.
    sqrt(k) has unit L2 norm for every kernel (k sums to 1), keeping inputs O(1).
    """
    k = kernels / kernels.sum((1, 2), keepdim=True)
    desc = torch.cat([torch.ones_like(sigma)[:, None], (sigma / 40)[:, None],
                      k.clamp_min(0).sqrt().flatten(1).to(sigma.dtype)], 1)
    if present is not None:
        desc = desc * present.to(desc.dtype)[:, None]
    return desc


# conditioning family -> (descriptor size, descriptor function)
DESCRIPTORS = {"params": (N_DESC, degradation_descriptor), "kernel": (N_KDESC, kernel_descriptor),
               "unrolled": (N_KDESC, kernel_descriptor), "unrolled_emb": (N_KDESC, kernel_descriptor),
               "unrolled_both": (N_KDESC, kernel_descriptor), "unrolled_blind": (N_KDESC, kernel_descriptor)}  # unrolled decodes k and sigma from it


def wiener_deconv(y, kernels, sigma, floor=0.0, alpha=300.0):
    """Physics-based conditioning: non-blind deconvolution with the given kernel and sigma.

    X = conj(H) Y / (|H|^2 + lam |G|^2), gradient (Tikhonov) prior, lam = alpha * s^2
    with s the total white-noise std on the 0-1 scale. Reflect-padded, circular FFT.
    y: (B,C,H,W); kernels: (B,K,K); sigma: (B,) in 0-255 units.
    """
    b, c, h, w = y.shape
    ks = kernels.shape[-1]
    p = 2 * ks
    yp = F.pad(y.float(), [p] * 4, mode="reflect")
    hp, wp = yp.shape[-2:]
    kp = torch.zeros(b, hp, wp, device=y.device)
    kp[:, :ks, :ks] = kernels.float()
    kp = torch.roll(kp, (-(ks // 2), -(ks // 2)), dims=(1, 2))  # kernel centre at origin
    H = torch.fft.rfft2(kp)[:, None]
    gx = torch.zeros(hp, wp, device=y.device)
    gx[0, 0], gx[0, 1] = -1, 1
    G2 = torch.fft.rfft2(gx).abs() ** 2 + torch.fft.rfft2(gx.T.contiguous()).abs() ** 2
    s2 = ((sigma.float() ** 2 + floor ** 2) / 255 ** 2)[:, None, None, None]
    X = H.conj() * torch.fft.rfft2(yp) / (H.abs() ** 2 + alpha * s2 * G2)
    x = torch.fft.irfft2(X, s=(hp, wp))[..., p:p + h, p:p + w]
    return x.clamp(-0.5, 1.5).to(y.dtype)


def model_input(cond, y, kernels, sigma, floor=0.0):
    """Network input: the observation, plus the Wiener estimate for cond='wiener'."""
    if cond == "wiener":
        return torch.cat([y, wiener_deconv(y, kernels, sigma, floor)], 1)
    return y


IN_CH = {"wiener": 6}  # input channels per conditioning family (default 3)

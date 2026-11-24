"""Noise level of the real Levin captures: oracle (residual against the sharp capture, as in the evaluation)
versus blind estimates from the blurred image alone (CPU only, no network).

  oracle : std of y - k*x at the best integer alignment within +-8 px, 25 px border (eval_shift.py)
  mad    : Donoho-Johnstone wavelet estimate, median(|HH1|)/0.6745 on the finest Haar diagonal subband
  immerkaer : Immerkaer (1996) fast noise variance estimator with the 3x3 Laplacian-difference mask
All values in grey levels (0-255). Writes reports/real_sigma.csv.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d

from ext_kernels import levin_real_pairs


def oracle(x, y, k):
    kx = convolve2d(x, k, mode="same", boundary="symm")
    return min(float((np.roll(kx, (dy, dx), (0, 1)) - y)[25:-25, 25:-25].std() * 255)
               for dy in range(-8, 9) for dx in range(-8, 9))


def mad(y):
    h, w = (y.shape[0] // 2) * 2, (y.shape[1] // 2) * 2
    a = y[:h, :w]
    hh = (a[0::2, 0::2] - a[0::2, 1::2] - a[1::2, 0::2] + a[1::2, 1::2]) / 2
    return float(np.median(np.abs(hh)) / 0.6745 * 255)


def immerkaer(y):
    m = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    r = convolve2d(y, m, mode="valid")
    h, w = y.shape
    return float(np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2)) * np.abs(r).sum() * 255)


def main():
    rows = []
    for name, x, y, k in levin_real_pairs():
        rows.append((name, oracle(x, y, k), mad(y), immerkaer(y)))
    out = Path("reports/real_sigma.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "oracle", "mad", "immerkaer"])
        w.writerows([(n, f"{a:.3f}", f"{b:.3f}", f"{c:.3f}") for n, a, b, c in rows])
    a = np.array([r[1:] for r in rows])
    print("mean oracle %.2f  mad %.2f  immerkaer %.2f" % tuple(a.mean(0)))
    print("range oracle %.2f-%.2f | mad %.2f-%.2f | immerkaer %.2f-%.2f" % (a[:, 0].min(), a[:, 0].max(), a[:, 1].min(),
                                                                        a[:, 1].max(), a[:, 2].min(), a[:, 2].max()))
    print("corr oracle-mad %.2f, oracle-immerkaer %.2f" % (np.corrcoef(a[:, 0], a[:, 1])[0, 1],
                                                           np.corrcoef(a[:, 0], a[:, 2])[0, 1]))
    print("mean |mad-oracle| %.2f, |imm-oracle| %.2f" % (np.abs(a[:, 1] - a[:, 0]).mean(), np.abs(a[:, 2] - a[:, 0]).mean()))


if __name__ == "__main__":
    main()

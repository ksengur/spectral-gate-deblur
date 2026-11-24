# Dated design log of the consistency gate

The order in which the gate's choices were fixed, as referenced in Section V-B ("What was fixed when").
The dates come from the project notes and the file times of the scripts.

| Date | Decision | Results seen when it was made |
|---|---|---|
| 2026-11-08 | Wiener baseline `degrade_torch.wiener_deconv`: gradient (Tikhonov) prior, `alpha = 300`, so lambda = 300 s^2. It was not tuned afterwards. | Gaussian (in distribution) and motion blur only |
| 2026-11-08 | `dc_gate.py`: residual score rho with the Wiener fallback. | Gaussian, motion |
| 2026-11-08 | `gate2.py`: thresholds are the in-distribution 99 % quantile (q = 0.99) on the grid sigma in {1,2,3,5,10,15,25}, interpolated in log sigma. Calibration uses 50 held-out DIV2K training images and 12 fresh Gaussian kernels with both orders. The 10-pixel border is the same as in the PSNR protocol. | Gaussian, motion |
| 2026-11-10 | Levin, defocus and the real Levin captures are added (`ext_kernels.py`, `eval_shift.py`). | All families |
| 2026-11-10 | `gate3.py`: the residual-only gate is evaluated on all families. It accepts failing data-step outputs on defocus and on real captures. | All families |
| 2026-11-10 | `gate4.py`: the null-space score nu is added. It reuses lambda, q, the grid and the border unchanged, and its thresholds are calibrated in distribution only. **Its form was motivated by the gate3 failures under shift.** | All families |
| 2026-11-10/11 | The 0.5 dB tolerance for false accepts and false rejects is set as a reporting threshold for the decision-quality table. The gate does not use it. | All results |
| 2026-11-16 | Sensitivity checks: q (Table ablation A4), lambda of nu at 100 / 300 / 900 s^2 (A5, `run_lam.ps1`), and decision tolerance at 0.25 / 0.5 / 1 / 2 dB (Sec. VI, Decision Quality). | All results |

The region W > 1/2 is used only to shade the spectrum figure; the gate uses the soft weight W.

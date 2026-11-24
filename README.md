# Spectral Failure Modes and Consistency Gating for Non-Blind Deblurring Under Kernel Shift

Code for the paper *Spectral Failure Modes and Consistency Gating for Non-Blind Deblurring Under Kernel Shift*
by Abdulkadir Sengur, Massimo Salvi, Yan Tian and Rajendra Acharya (submitted to IEEE Transactions on Image
Processing).

The repository contains the factorial study of kernel-injection routes in an unrolled half-quadratic-splitting
network, the two-sided consistency gate (re-blurring residual ρ and near-null-space score ν with a Wiener fallback),
and the scripts that regenerate every table and figure of the paper.

## Contents

| Path | What it does |
|---|---|
| `degrade_torch.py` | Blur/noise formation (both orders), kernel sets, Wiener deconvolution |
| `nafnet.py`, `unrolled.py` | NAFNet prior and unrolled HQS network (arms: `unrolled`, `unrolled_emb`, `unrolled_both`, `unrolled_blind`) |
| `train_2x2.py`, `eval_2x2.py` | Training and in-distribution evaluation |
| `ext_kernels.py`, `eval_shift.py` | Shifted kernel families (motion, Levin, defocus) and the real Levin et al. captures |
| `gate3.py`, `gate4.py` | Residual-only gate and the two-sided gate (calibration on in-distribution data only) |
| `eval_pretrained.py` | USRNet and DPIR with official weights under the same protocol |
| `ssim_eval.py`, `real_sigma.py` | SSIM evaluation and blind noise estimates for the real captures |
| `dc_gate.py`, `gate2.py` | Earlier gate versions, kept because the design log refers to them |
| `analyze_*.py` | Paired, image-level bootstrap analyses |
| `scripts/*.ps1` | Training and evaluation runs of the paper (PowerShell; each line is a plain `python` call) |
| `paper/make_tables.py`, `paper/make_figures.py`, `paper/extra_analysis.py` | Tables, figures and supplementary analyses |
| `reports/design_log.md` | Dated log of when each gate choice was fixed |
| `paths.py` | Data and output locations (environment variables, see below) |

## Setup

Python 3.10 or later and PyTorch 2.x (a CUDA GPU is recommended for training).

```bash
pip install -r requirements.txt
```

Data (not redistributed here):

- DIV2K: <https://data.vision.ee.ethz.ch/cvl/DIV2K/>, `DIV2K_train_HR` and `DIV2K_valid_HR` under `data/DIV2K/`.
- Levin et al. (2009) real blurred captures (`imXX_flitYY.mat`) under `data/levin09/Levin09blurdata/`.
- Only for `eval_pretrained.py`: the KAIR code (<https://github.com/cszn/KAIR>) under `third_party/KAIR/` and the
  official `usrnet.pth` and `drunet_color.pth` weights under `pretrained/`.

Other locations can be set with the environment variables `DIV2K_DIR`, `LEVIN_DIR`, `KAIR_DIR`, `PRETRAINED_DIR`
and `GATE_OUT` (gate outputs, default `outputs/`).

## Reproducing the paper

### Tables from the released outputs (no GPU, no data needed)

Download `outputs.zip` from the release of this repository, extract it in the repository root (it creates `runs/`,
`runs_cond/` and `outputs/`), then run

```bash
python paper/make_tables.py
```

This writes `paper/tables/*.tex` and `reports/paper_numbers.md`; the tables are identical to those in the paper.
`python paper/make_figures.py` regenerates the figures (some panels need DIV2K and the checkpoints).

### Gate evaluation from the released checkpoints

Extract `checkpoints.zip` from the release in the repository root, then, for example,

```bash
python gate4.py --runs runs_cond/unrolled_s0 runs_cond/unrolled_emb_s0 \
    --train-dir data/DIV2K/DIV2K_train_HR --test-dir data/DIV2K/DIV2K_valid_HR
```

The thresholds are calibrated on 50 held-out DIV2K training images with fresh Gaussian kernels only; no test image,
shifted kernel or real capture is used for calibration.

### Training from scratch

```powershell
.\scripts\run_factorial.ps1 -Train data\DIV2K\DIV2K_train_HR -Valid data\DIV2K\DIV2K_valid_HR -Seeds 0,1,2
.\scripts\run_mixk.ps1 -Seeds 0          # second training distribution (Gaussian + motion kernels)
.\scripts\run_lam.ps1                     # sensitivity to the weight of the nu score (ablation A5)
.\scripts\run_realblind.ps1               # blind noise estimates on the real captures
```

Seeds, iteration counts and all hyper-parameters are the defaults of `train_2x2.py` and the values in the scripts.

## Licence

Code: MIT (see `LICENSE`). DIV2K, the Levin et al. data and the USRNet/DPIR weights are subject to their own licences
and are not included.

## Citation

```bibtex
@article{spectralgate2026,
  title   = {Spectral Failure Modes and Consistency Gating for Non-Blind Deblurring Under Kernel Shift},
  author  = {Sengur, Abdulkadir and Salvi, Massimo and Tian, Yan and Acharya, Rajendra},
  journal = {IEEE Transactions on Image Processing},
  note    = {submitted},
  year    = {2026}
}
```

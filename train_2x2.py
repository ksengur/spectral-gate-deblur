"""Train one cell of the 2x2 order-mismatch study.

Runs (each with 3 seeds):
  --train-order bn  --cond none    blind expert for blur->noise
  --train-order nb  --cond none    blind expert for noise->blur
  --train-order mix --cond none    blind model trained on both orders
  --train-order mix --cond oracle  same architecture, receives the true order

All runs use the identical NAFNet (FiLM layers exist in every model; blind
models always receive the UNKNOWN token), so parameter counts are equal.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from degrade_torch import (DESCRIPTORS, IN_CH, MODES, degrade_pair, eval_kernel_sets, model_input,
                          motion_kernel, random_gaussian_kernel)
from nafnet import BLUR_THEN_NOISE, NOISE_THEN_BLUR, UNKNOWN, NAFNet
from unrolled import build_model

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_cache(image_dir, cache_dir):
    """Decode each PNG once into an uncompressed .npy so workers can memory-map it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not files:
        raise SystemExit(f"No images in {image_dir}")
    out = []
    for i, path in enumerate(files):
        target = cache_dir / (path.stem + ".npy")
        if not target.exists():
            with Image.open(path) as im:
                np.save(target, np.asarray(im.convert("RGB"), dtype=np.uint8))
            if i % 100 == 0:
                print(f"cache {i}/{len(files)}", flush=True)
        out.append(target)
    return out


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, files, crop, length, seed):
        self.files, self.crop, self.length, self.seed = files, crop, length, seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, idx])
        img = np.load(self.files[rng.integers(len(self.files))], mmap_mode="r")
        h, w = img.shape[:2]
        top, left = rng.integers(h - self.crop + 1), rng.integers(w - self.crop + 1)
        patch = np.array(img[top:top + self.crop, left:left + self.crop])
        patch = np.rot90(patch, rng.integers(4))
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1).float() / 255


def center_crops(files, size):
    crops = []
    for f in files:
        img = np.load(f, mmap_mode="r")
        h, w = img.shape[:2]
        t, l = (h - size) // 2, (w - size) // 2
        crops.append(np.array(img[t:t + size, l:l + size]))
    return torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255


def psnr(pred, target):
    mse = (pred.clamp(0, 1) - target).pow(2).flatten(1).mean(1)
    return -10 * torch.log10(mse.clamp_min(1e-12))


@torch.no_grad()
def dev_eval(model, dev, args, device):
    """Small in-distribution check on the dev split (not used for model selection)."""
    model.eval()
    kernels = torch.from_numpy(eval_kernel_sets()["gauss"]).float()
    gen = torch.Generator(device=device).manual_seed(999)
    scores = {0: [], 1: []}
    for i in range(0, len(dev), 8):
        x = dev[i:i + 8].to(device)
        for kid in range(len(kernels)):
            k = kernels[kid:kid + 1].expand(len(x), -1, -1).to(device)
            sigma = torch.full((len(x),), 15.0, device=device)
            pair = degrade_pair(x, k, sigma, args.mode, args.floor, gen)
            params = DESCRIPTORS[args.cond][1](k, sigma) if args.cond in DESCRIPTORS else None
            for order, y in enumerate(pair):
                tok = torch.full((len(x),), order if args.cond == "oracle" else UNKNOWN,
                                 device=device, dtype=torch.long)
                scores[order].append(psnr(model(model_input(args.cond, y, k, sigma, args.floor), tok, params).float(), x))
    model.train()
    return {o: torch.cat(v).mean().item() for o, v in scores.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", type=Path, required=True, help="DIV2K_train_HR (0001-0800)")
    p.add_argument("--cache", type=Path, default=Path("cache/train"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-order", choices=("bn", "nb", "mix"), required=True)
    p.add_argument("--cond", choices=("none", "oracle", "params", "kernel", "wiener", "unrolled", "unrolled_emb", "unrolled_both",
                                     "unrolled_blind"), required=True,
                   help="params: [sigma, kernel moments]; kernel: [sigma, full kernel]; wiener: Wiener estimate as input; no order")
    p.add_argument("--token-dropout", type=float, default=0.0,
                   help="probability of hiding the conditioning signal during training")
    p.add_argument("--mode", choices=MODES, default="matched_effective_energy")
    p.add_argument("--floor", type=float, default=0.29, help="post-blur noise floor, 0-255 units")
    p.add_argument("--sigma-range", type=float, nargs=2, default=(1, 40))
    p.add_argument("--train-kernels", choices=("gauss", "mix"), default="gauss",
                   help="mix: Gaussian + random motion kernels (second training distribution)")
    p.add_argument("--p-motion", type=float, default=0.5, help="share of motion kernels when --train-kernels mix")
    p.add_argument("--width-range", type=float, nargs=2, default=(0.2, 3.0))
    p.add_argument("--dev-count", type=int, default=50, help="last N train images held out as dev")
    p.add_argument("--iters", type=int, default=50000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--dev-every", type=int, default=5000)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    if args.cond != "none" and args.train_order != "mix":
        p.error("conditioning is only meaningful with --train-order mix")
    if not 0 <= args.token_dropout < 1 or (args.cond == "none" and args.token_dropout):
        p.error("token dropout must be in [0, 1) and needs a conditioned model")

    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng([args.seed, 1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    args.out.mkdir(parents=True, exist_ok=True)
    # Plain types only, so the checkpoint loads with torch.load(weights_only=True).
    plain_args = {k: str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v
                  for k, v in vars(args).items()}
    (args.out / "config.json").write_text(json.dumps(plain_args, indent=2))

    files = build_cache(args.train_dir, args.cache)
    train_files, dev_files = files[:-args.dev_count], files[-args.dev_count:]
    print(f"train {len(train_files)} images, dev {len(dev_files)} images "
          f"({dev_files[0].stem}-{dev_files[-1].stem})", flush=True)
    dev = center_crops(dev_files, 256)

    data = CropDataset(train_files, args.crop, args.iters * args.batch, args.seed)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, shuffle=False,
                                         num_workers=args.workers, pin_memory=True,
                                         drop_last=True, persistent_workers=args.workers > 0)
    model = build_model(args.cond, args.width, DESCRIPTORS[args.cond][0] if args.cond in DESCRIPTORS else 0,
                        IN_CH.get(args.cond, 3), args.floor).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params / 1e6:.3f} M on {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=1e-7)
    use_amp = device.type == "cuda" and not args.no_amp

    log = open(args.out / "train_log.csv", "w", encoding="utf-8")
    if log.tell() == 0:
        log.write("iter,loss,lr,it_per_s,dev_psnr_bn,dev_psnr_nb\n")
    t0, running = time.time(), 0.0
    model.train()
    for it, x in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        b = x.shape[0]
        if args.train_kernels == "gauss":  # original random stream, unchanged
            kernels = np.stack([random_gaussian_kernel(np_rng, args.width_range) for _ in range(b)])
        else:  # Gaussian + motion mixture; motion drawn from the training stream, not the locked eval seed
            kernels = np.stack([motion_kernel(np_rng) if np_rng.random() < args.p_motion
                                else random_gaussian_kernel(np_rng, args.width_range) for _ in range(b)])
        kernels = torch.from_numpy(kernels).float().to(device)
        sigma = torch.from_numpy(np_rng.uniform(*args.sigma_range, b)).float().to(device)
        if args.train_order == "bn":
            order = torch.zeros(b, dtype=torch.long, device=device)
        elif args.train_order == "nb":
            order = torch.ones(b, dtype=torch.long, device=device)
        else:
            order = torch.from_numpy(np_rng.integers(0, 2, b)).to(device)
        bn, nb = degrade_pair(x, kernels, sigma, args.mode, args.floor)
        y = torch.where((order == NOISE_THEN_BLUR)[:, None, None, None], nb, bn)
        # draw only when needed, so dropout-free runs keep the original random stream
        keep = (torch.from_numpy(np_rng.random(b) >= args.token_dropout).to(device) if args.token_dropout
                else torch.ones(b, dtype=torch.bool, device=device))
        token = torch.where(keep, order, UNKNOWN) if args.cond == "oracle" else torch.full_like(order, UNKNOWN)
        params = DESCRIPTORS[args.cond][1](kernels, sigma, keep) if args.cond in DESCRIPTORS else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            pred = model(model_input(args.cond, y, kernels, sigma, args.floor), token, params)
        loss = F.mse_loss(pred.float(), x)  # L2: matches the MMSE analysis
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running += loss.item()
        if it % args.log_every == 0 or it == args.iters:
            dev_scores = dev_eval(model, dev, args, device) if (it % args.dev_every == 0 or it == args.iters) else None
            speed = it / (time.time() - t0)
            line = (f"{it},{running / args.log_every:.6g},{sched.get_last_lr()[0]:.3g},{speed:.2f},"
                    + (f"{dev_scores[0]:.3f},{dev_scores[1]:.3f}" if dev_scores else ","))
            log.write(line + "\n"); log.flush()
            eta = (args.iters - it) / speed / 3600
            print(f"it {it}/{args.iters} loss {running / args.log_every:.5f} {speed:.1f} it/s "
                  f"eta {eta:.2f} h" + (f" dev BN {dev_scores[0]:.2f} NB {dev_scores[1]:.2f}" if dev_scores else ""),
                  flush=True)
            running = 0.0
            torch.save({"model": model.state_dict(), "iter": it, "args": plain_args}, args.out / "last.pt")
    log.close()


if __name__ == "__main__":
    main()

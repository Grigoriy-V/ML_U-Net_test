from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset

from mini_diffusion.diffusion import EMA
from mini_diffusion.sit import linear_interpolant, velocity_loss
from mini_diffusion.train_sit import build_model, build_optimizer, device_dtype, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Short synthetic SiT-B/2 compute benchmark; never reads or writes AFHQ cache/checkpoints.")
    parser.add_argument("--config", required=True); parser.add_argument("--batch", type=int, required=True); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10); parser.add_argument("--steps", type=int, default=30); parser.add_argument("--output", required=True)
    args = parser.parse_args(); cfg = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("AFHQ SiT-B/2 benchmark requires CUDA")
    cfg["data"]["batch_size"] = args.batch; cfg["data"]["num_workers"] = args.workers
    device, dtype, autocast = device_dtype(cfg); torch.backends.cudnn.benchmark = bool(cfg.get("performance", {}).get("cudnn_benchmark", False))
    samples = max(args.batch * 4, 1024)
    dataset = TensorDataset(torch.randn(samples, 4, 16, 16), torch.zeros(samples, dtype=torch.long))
    kwargs = {"batch_size": args.batch, "shuffle": True, "drop_last": True, "num_workers": args.workers, "pin_memory": True}
    if args.workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(dataset, **kwargs); iterator = iter(loader)
    model = build_model(cfg).to(device); optimizer = build_optimizer(model, cfg); ema = EMA(model, cfg["train"]["ema_decay"], foreach=cfg["performance"].get("ema_foreach", False))
    def update() -> None:
        nonlocal iterator
        try: x0, labels = next(iterator)
        except StopIteration: iterator = iter(loader); x0, labels = next(iterator)
        x0, labels = x0.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True); noise, t = torch.randn_like(x0), torch.rand(x0.shape[0], device=device); xt, target = linear_interpolant(x0, noise, t)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast): loss = velocity_loss(model(xt, t, labels), target)
        loss.backward(); optimizer.step(); ema.update(model)
    result = {"batch": args.batch, "workers": args.workers, "warmup_steps": args.warmup, "measured_steps": args.steps, "synthetic_latents": True, "fused_adamw": bool(optimizer.defaults.get("fused", False)), "foreach_ema": bool(cfg["performance"].get("ema_foreach", False)), "oom": False}
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.warmup): update()
        torch.cuda.synchronize(device); started = time.perf_counter()
        for _ in range(args.steps): update()
        torch.cuda.synchronize(device); seconds = (time.perf_counter() - started) / args.steps
        result.update({"images_per_second": args.batch / seconds, "ms_per_step": seconds * 1000, "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3, "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3})
    except torch.OutOfMemoryError:
        result["oom"] = True; torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

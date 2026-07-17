from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import torch
from mini_diffusion.sit import linear_interpolant, velocity_loss
from mini_diffusion.train_sit import build_model, build_optimizer, device_dtype, load_config


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--batches", nargs="+", type=int, default=[64, 128, 256]); parser.add_argument("--warmup", type=int, default=2); parser.add_argument("--steps", type=int, default=5); parser.add_argument("--output", default="reports/imagenette_sit_benchmark.json"); args = parser.parse_args()
    cfg = load_config(args.config); device, dtype, autocast = device_dtype(cfg); results = []
    for batch in args.batches:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device); model = build_model(cfg).to(device); opt = build_optimizer(model, cfg); x0 = torch.randn(batch, 4, 16, 16, device=device); labels = torch.arange(batch, device=device) % 10
            def update():
                opt.zero_grad(set_to_none=True); t = torch.rand(batch, device=device); noise = torch.randn_like(x0); xt, target = linear_interpolant(x0, noise, t)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast): loss = velocity_loss(model(xt, t, labels), target)
                loss.backward(); opt.step()
            for _ in range(args.warmup): update()
            torch.cuda.synchronize(device) if device.type == "cuda" else None; start = time.perf_counter()
            for _ in range(args.steps): update()
            torch.cuda.synchronize(device) if device.type == "cuda" else None; seconds = (time.perf_counter() - start) / args.steps
            results.append({"batch": batch, "effective_batch": batch * cfg["train"].get("grad_accum_steps", 1), "seconds_per_step": seconds, "images_per_second": batch / seconds, "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0, "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0, "oom": False})
        except torch.OutOfMemoryError:
            results.append({"batch": batch, "oom": True}); torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8"); print(json.dumps(results, indent=2))


if __name__ == "__main__": main()

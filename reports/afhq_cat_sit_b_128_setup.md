# AFHQ Cats SiT-B/2 128x128 Setup

Date: 2026-07-18. This is an isolated one-class, non-REPA experiment. Long training was not started.

## Design

- Dataset: official AFHQ cats only. Expected local layout is `datasets/afhq/train/cat` plus `datasets/afhq/test/cat`; the official StarGAN v2 `val/cat` held-out layout is accepted as a read-only test alias.
- Training never opens the test cache. `prepare_afhq_cat_latents.py` writes separate `train.pt` and `test.pt` caches and JSONL manifests under `outputs/afhq_cat_sit_b_128/latents/`.
- Train cache: four deterministic `RandomResizedCrop(scale=0.85-1.0, ratio=1.0)` plus horizontal-flip variants per source image. Each manifest row records source-relative path, augmentation seed, split, source SHA-256, pixel SHA-256, and latent SHA-256. Cache creation fails if four variants of one source are not pixel-distinct.
- Test cache: one deterministic center-square crop per source image, without training augmentation.
- VAE: `stabilityai/sd-vae-ft-mse`; scaling factor is read from VAE config and stored in cache metadata.
- Model: SiT-B/2 velocity model, 128x128 image resolution, 16x16 latent resolution, 4 latent channels, one class plus the CFG null token, hidden size 768, depth 12, 12 heads, patch size 2, MLP ratio 4.0. Trainable parameters: `129,929,488`.
- REPA, DINO teacher, projector, feature cache, and REPA loss are absent.

## Dataset Status

`datasets/afhq/` is absent at setup time, so actual official cat train/test counts and cache counts are not yet available. After placement, the report must record the manifest-derived counts: train cache is exactly `4 * train source images`; test cache is exactly `1 * held-out test source images`. The loader accepts only `train/cat` for training and only `test/cat` (or official `val/cat` alias) for held-out evaluation.

When both caches are created, `prepare_afhq_cat_latents.py` writes `reports/afhq_cat_sit_b_128_cache_stats.md` with the real source counts, cached latent counts, augmentation variants, and cache fingerprints. Add that generated small report to the follow-up cache-preparation commit.

## Training Plan

The full config is `mini_diffusion/configs/afhq_cat_sit_b_128.yaml`.

- AdamW, BF16, fused AdamW when available, foreach EMA, LR `1e-4`.
- Linear warmup for 1,000 optimizer steps, then cosine decay to `1e-5` over independent `scheduler_total_steps: 100000`. `--max-steps 10000` is only a stop-at limit and does not compress the schedule. Scheduler state is written to every checkpoint and restored on resume.
- Maximum cap: 100k optimizer steps; `latest.pt` every 10k and immutable `step_0010000.pt` through `step_0100000.pt` milestones.
- Fixed decoded raw and EMA previews every 5k: the same initial noise seed, Heun-25, CFG 1.0 and 1.5.
- Training should stop when fixed previews and held-out metrics cease improving; 100k is a cap, not a required target.

## Performance Check

Synthetic latent compute benchmark on RTX 4090, PyTorch `2.11.0+cu128`, BF16, 10 warmup + 30 measured updates, workers 4. It exercises SiT-B/2, optimizer and EMA but does not measure AFHQ cache/JPEG IO.

| Physical batch | Images/s | ms/step | Peak allocated GB | Peak reserved GB | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 2072.26 | 61.77 | 5.77 | 6.19 | Stable |
| 256 | 2030.98 | 126.05 | 9.27 | 9.70 | Stable, slower |

Both runs used fused AdamW and foreach EMA. Choose physical batch `128` with `grad_accum_steps: 2`, retaining effective batch `256` while being slightly faster and leaving more VRAM headroom. Raw JSON: `reports/afhq_cat_sit_b_128_benchmark_batch128.json` and `reports/afhq_cat_sit_b_128_benchmark_batch256.json`.

## Checks Run

```powershell
.\.venv\Scripts\python.exe -m pytest mini_diffusion\tests -q
.\.venv\Scripts\python.exe mini_diffusion\benchmark_afhq_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128.yaml --batch 128 --workers 4 --warmup 10 --steps 30 --output reports\afhq_cat_sit_b_128_benchmark_batch128.json
.\.venv\Scripts\python.exe mini_diffusion\benchmark_afhq_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128.yaml --batch 256 --workers 4 --warmup 10 --steps 30 --output reports\afhq_cat_sit_b_128_benchmark_batch256.json
```

Result: `37 passed`. Unit coverage includes official train/test isolation, deterministic four-variant manifests, generic one-batch velocity overfit, deterministic Euler/Heun sampling, scheduler checkpoint/resume state, and the separate CLI stop-at versus scheduler-horizon contract.

AFHQ cache preparation, AFHQ one-batch overfit, train/checkpoint/resume/sample smoke, finite raw/EMA check on a produced AFHQ checkpoint, VAE reconstruction ceiling, and AFHQ FID/KID evaluation were not run because the official AFHQ data is absent from `datasets/afhq/`. The loader failed with the expected clear error and did not create a cache or checkpoint.

## Final Pre-10k Preflight

- `training_limits(config, cli_max_steps=10000)` returned `(10000, 100000)`: the CLI stops this invocation at 10k while the warmup/cosine scheduler retains the configured 100k horizon.
- Train augmentation now follows deterministic `RandomResizedCrop(scale=0.85-1.0, ratio=1.0)` with alternate horizontal flip. During real cache creation, pixel and latent SHA-256 are recorded for every variant and any source with fewer than four unique train pixel variants causes a hard failure.
- No cache, VAE ceiling, IO benchmark, overfit, train/resume/sample smoke, or long run was executed without real AFHQ images.

## First Runtime Chain After Dataset Placement

The official source is [clovaai/stargan-v2 AFHQ instructions](https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq). Extract to `datasets/afhq/` with `train/cat` and `test/cat` (or held-out `val/cat`). The official dataset is CC BY-NC 4.0; comply with its license before use.

```powershell
.\.venv\Scripts\python.exe mini_diffusion\prepare_afhq_cat_latents.py --config mini_diffusion\configs\afhq_cat_sit_b_128_debug.yaml --limit 32
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128_debug.yaml --overfit-smoke --overfit-updates 40
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128_debug.yaml
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128_debug.yaml --resume outputs\afhq_cat_sit_b_128_debug\checkpoints\latest.pt --max-steps 3
.\.venv\Scripts\python.exe mini_diffusion\sample_sit.py --checkpoint outputs\afhq_cat_sit_b_128_debug\checkpoints\latest.pt --weights ema --classes 0 0 0 0 --seeds 1000 1001 1002 1003 --sampler heun --steps 25 --guidance-scale 1.5 --output outputs\afhq_cat_sit_b_128_debug\samples
```

Then build the full cache, measure real cache/loader IO for both physical candidates, and only then begin the first bounded 10k run:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\prepare_afhq_cat_latents.py --config mini_diffusion\configs\afhq_cat_sit_b_128.yaml
.\.venv\Scripts\python.exe mini_diffusion\benchmark_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128.yaml --cache-path outputs\afhq_cat_sit_b_128\latents\train.pt --batch 128 --workers 4 --warmup 10 --steps 50 --output reports\afhq_cat_sit_b_128_real_cache_batch128.json
.\.venv\Scripts\python.exe mini_diffusion\benchmark_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128.yaml --cache-path outputs\afhq_cat_sit_b_128\latents\train.pt --batch 256 --workers 4 --warmup 10 --steps 50 --output reports\afhq_cat_sit_b_128_real_cache_batch256.json
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128.yaml --max-steps 10000
```

After a checkpoint exists, evaluate the held-out test split and VAE ceiling without ResNet class metrics:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\evaluate_afhq_cat.py --checkpoint outputs\afhq_cat_sit_b_128\checkpoints\step_0010000.pt --output evaluation\afhq_cat_sit_b_128\raw_cfg1_0 --weights raw --samples 1000 --steps 50 --guidance-scale 1.0
.\.venv\Scripts\python.exe mini_diffusion\evaluate_afhq_cat.py --checkpoint outputs\afhq_cat_sit_b_128\checkpoints\step_0010000.pt --output evaluation\afhq_cat_sit_b_128\raw_cfg1_5 --weights raw --samples 1000 --steps 50 --guidance-scale 1.5
.\.venv\Scripts\python.exe mini_diffusion\evaluate_afhq_cat.py --checkpoint outputs\afhq_cat_sit_b_128\checkpoints\step_0010000.pt --output evaluation\afhq_cat_sit_b_128\ema_cfg1_0 --weights ema --samples 1000 --steps 50 --guidance-scale 1.0
.\.venv\Scripts\python.exe mini_diffusion\evaluate_afhq_cat.py --checkpoint outputs\afhq_cat_sit_b_128\checkpoints\step_0010000.pt --output evaluation\afhq_cat_sit_b_128\ema_cfg1_5 --weights ema --samples 1000 --steps 50 --guidance-scale 1.5
.\.venv\Scripts\python.exe mini_diffusion\evaluate_afhq_cat.py --checkpoint outputs\afhq_cat_sit_b_128\checkpoints\step_0010000.pt --output evaluation\afhq_cat_sit_b_128\vae_ceiling_test --vae-ceiling --sample-batch-size 32
```

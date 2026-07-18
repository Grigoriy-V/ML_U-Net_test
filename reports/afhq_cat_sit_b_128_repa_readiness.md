# AFHQ Cats SiT-B/2 REPA Readiness

Date: 2026-07-18. This is a separate REPA experiment that starts from a new initialization. The non-REPA AFHQ baseline and its checkpoints were not modified.

## Configuration

- Config: `mini_diffusion/configs/afhq_cat_sit_b_128_repa.yaml` (SHA-256 `845D089F479BBE405FD127AEFAFCE372E7E5775D6004E296E29A19B561D7563B`).
- Output directory: `outputs/afhq_cat_sit_b_128_repa/`.
- Input cache: the existing baseline train cache, fingerprint `c94251a82755ce2c1e02269404b3d5b92b5f2325fcefe51e53694c316b4e5612`; it is read-only.
- SiT-B/2: latent 16x16, patch size 2, hidden size 768, depth 12, 12 heads, MLP ratio 4.0, one class plus CFG null token.
- Physical batch 128, accumulation 2, effective batch 256; AdamW `1e-4`, warmup 1,000, cosine horizon 100,000, minimum LR `1e-5`, BF16, fused AdamW, foreach EMA.
- The SiT-only initial parameter vector is SHA-256 `ca4fbfbaf6cb4a88a16b23e563739b27faee6c9033014ca77119b53e8f655a97` for both baseline and REPA config under seed 123. Projector construction restores RNG state, so adding it does not perturb SiT initialization.

## REPA

- Teacher: frozen `facebookresearch/dinov2`, `dinov2_vitb14`, revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, eval mode and no gradients.
- Teacher input: the exact deterministic cached AFHQ crop/flip, RGB128 to bicubic-antialiased 224, then ImageNet normalization.
- Features: 64 tokens x 768, DINO 16x16 patch grid adaptively pooled to 8x8.
- Student connection: output after SiT block 8 (`alignment_depth: 8`).
- Projector: `768 -> 2048 -> 2048 -> 768` with SiLU; 7,344,896 trainable parameters. Total trainable parameters: 137,274,384, including 129,929,488 SiT parameters.
- REPA loss is negative tokenwise cosine; coefficient 0.5, constant for the full run. Teacher features are precomputed and memory-mapped, not loaded during training.

## Teacher Feature Cache

Command actually run:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\prepare_repa_features.py --config mini_diffusion\configs\afhq_cat_sit_b_128_repa.yaml --split train
```

`outputs/afhq_cat_sit_b_128_repa/dino_features/train/` contains 20,612 finite FP16 grids, fingerprint `593f73814354ba735199b68a9d84185b7a797cba4605dd0a48ad23d2198b116a`, size 2,026,242,176 bytes. The feature-cache metadata records the AFHQ train-manifest SHA-256 `a92dd98ff1197614d4d1a737994f0c40dc1394c80a55f68f9d605aa7a8ebbfb2`; preparation reconstructs and compares source path, augmentation seed, augmentation index, split, and source hash before extracting each feature. This prevents mixing a latent with a feature from another crop/flip variant.

## Logging And Runtime Checks

`train_sit.py` now accumulates detached per-microbatch values separately from the scaled backward loss. Each optimizer-step TensorBoard record is the mean over all microbatches:

- `train/loss`: mean total objective;
- `train/flow_loss`: mean flow objective;
- `train/repa_loss`: mean unweighted REPA objective;
- `train/repa_weighted_loss`: mean coefficient-weighted REPA contribution.

When REPA is disabled, no REPA tags are written and `train/loss` equals the averaged `train/flow_loss`. Gradient scaling and optimizer semantics are unchanged.

Commands actually run:

```powershell
.\.venv\Scripts\python.exe -m pytest mini_diffusion\tests -q
.\.venv\Scripts\python.exe mini_diffusion\prepare_repa_features.py --config mini_diffusion\configs\afhq_cat_sit_b_128_repa_debug.yaml --split train --limit 8
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128_repa_debug.yaml --overfit-smoke --overfit-updates 40
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128_repa_debug.yaml --max-steps 2
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128_repa_debug.yaml --resume outputs\afhq_cat_sit_b_128_repa_debug\checkpoints\latest.pt --max-steps 3
.\.venv\Scripts\python.exe mini_diffusion\benchmark_repa.py --config mini_diffusion\configs\afhq_cat_sit_b_128_repa.yaml --mode repa --warmup 10 --steps 50 --output reports\afhq_cat_sit_b_128_repa_benchmark.json
```

- `pytest`: 38 passed.
- One-batch REPA overfit: total loss `1.447892 -> 0.333109`; cosine `0.431317 -> 0.733416` over 40 updates.
- Actual CUDA BF16 train -> checkpoint -> resume: step 2 wrote `latest.pt` and immutable debug milestone `step_0000002.pt`; resume reached step 3. The checkpoint contained model, EMA, optimizer, scheduler, REPA projector, feature-cache metadata, and RNG states. Decoded raw and EMA Heun-2 PNG previews were finite.
- RTX 4090, CUDA 12.8, PyTorch 2.11.0+cu128: REPA benchmark (10 warmup plus 50 measured updates) reached 1,942.69 images/s, 65.89 ms/step, 5.59 GB peak allocated, and 5.93 GB peak reserved. Fused AdamW and memory-efficient SDP attention were active.
- The baseline frozen checkpoint remains unchanged: `outputs/afhq_cat_sit_b_128/checkpoints/best_raw_0020000.pt`, SHA-256 `1eb7db8e91d7727528421d09ddd82eeb8ca37573b12652f43c2180942a93e7f7`.

## Launch Decision

The isolated REPA run is ready to start at step 0. It will write separate `latest.pt`, immutable milestones, and decoded fixed raw/EMA previews at 5k, 10k, 15k, and 20k. No full 1,000-sample evaluation or sampler ablation has been launched.

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128_repa.yaml --max-steps 20000
```

This stop limit is 20,000 only; the scheduler horizon remains 100,000 from the config. Do not pass `--resume` or an `init_from` checkpoint. Stop the run if loss becomes non-finite, the projector loses gradients, or the latent/feature manifest validation fails.

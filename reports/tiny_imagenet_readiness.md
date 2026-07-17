# Tiny ImageNet 64x64 Readiness

## Status

The Tiny ImageNet archive is not present at `datasets/tiny-imagenet-200/`. No network download was attempted and no real Tiny ImageNet training was run. The real loader path was checked for fast, clear missing-data behavior and was exercised with a temporary miniature copy of the original directory structure.

The full model training path was measured on synthetic 64x64 inputs to decide whether physical batch 128 is suitable for the first run.

## Environment

- Python: 3.12.4
- PyTorch: 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 4090, 23.99 GB
- Dtype: BF16 autocast
- Trainable parameters: 48,371,755
- Model: `mini_diffusion/configs/tiny_imagenet.yaml`

## Batch And Attention Probe

Each comparison used the same U-Net, diffusion loss, gradient clipping, fused AdamW, foreach EMA, cuDNN autotuning, and effective batch 128. Measurements used two warmup optimizer steps and six synchronized measured optimizer steps in isolated processes. Inputs were synthetic GPU tensors, so the results exclude JPEG decoding and DataLoader throughput.

| Physical batch | Accumulation | Attention | Images/s | Mean optimizer step | Peak allocated | Peak reserved | Result |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 64 | 2 | manual | 462.85 | 0.2765 s | 7.73 GB | 8.29 GB | Keep |
| 128 | 1 | manual | 448.26 | 0.2855 s | 14.23 GB | 16.00 GB | Fits, reject as default |
| 64 | 2 | SDPA | 442.50 | 0.2893 s | 7.81 GB | 8.64 GB | Reject |

A separate first-step batch-128 probe also completed successfully at 13.97 GB peak allocated and 14.87 GB peak reserved. Its 33-second first step included cuDNN algorithm selection and is not a steady-state throughput measurement. All measured losses were finite.

## Decision

- Keep physical `batch_size: 64` with `grad_accum_steps: 2` for effective batch 128.
- Keep manual attention. SDPA was slower at this resolution and model shape.
- Enable pinned non-blocking transfer, four persistent workers, prefetch factor two, fused AdamW fallback, foreach EMA, cuDNN autotuning, and DDIM-50 periodic previews in the initial full config.
- Treat four workers as a starting value, not a final loader result. Re-benchmark workers and physical batch size with real JPEG files before starting the full run.
- Use `mini_diffusion/validate_tiny_imagenet.py` as the required data-integrity gate after extraction.

## Dataset Gate

The validator requires the original 200 classes, 100,000 training images, 10,000 validation images, 500 training images per class, and 50 validation images per class. It also checks annotations, one decoded DataLoader batch, `[3, 64, 64]` tensors, finite values, `[-1, 1]` normalization, and class indices.

Archive source: `https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip?download=1`

Expected archive MD5: `90528d7ca1a48142e341f4ef8d21d0de`

## Remaining Checks

After the archive is extracted:

1. Run `mini_diffusion/validate_tiny_imagenet.py` with `num_workers: 0` behavior.
2. Run the 64x64 one-batch overfit smoke test on real images.
3. Complete debug train, checkpoint, resume, and DDIM sample PNG.
4. Benchmark real-loader worker counts and compare `64 x 2` against `128 x 1` including JPEG decoding.
5. Start full training only after all four checks pass.

## Checks Actually Run

- `python -m pytest mini_diffusion/tests -q`: 16 passed.
- `python -m compileall -q mini_diffusion`: passed.
- `python mini_diffusion/validate_tiny_imagenet.py`: stopped immediately with the expected missing-dataset error and download URL.
- Full batch-128 CUDA training-step probe: passed with finite loss.
- Isolated synthetic comparisons for manual `64 x 2`, manual `128 x 1`, and SDPA `64 x 2`: passed with finite losses; results are in the table above.
- `benchmark.py` synthetic smoke with the prepared config and four persistent Windows workers: passed with finite loss and effective batch 128.
- Tiny ImageNet debug-model one-batch overfit on 64x64 FakeData: average loss decreased from 0.868714 in the first quarter to 0.362374 in the last quarter over 40 updates.

CUDA was used for every model/batch probe. Only synthetic smoke training and performance measurements were run; no Tiny ImageNet debug or full training run was performed.

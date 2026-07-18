# Operations guide (historical local commands)

This guide preserves practical commands from the original README. It is intentionally separate from the portfolio narrative. Commands reflect a Windows/PowerShell-oriented local workflow and **were not re-run during portfolio packaging**. They may require an RTX-class CUDA environment, local datasets/checkpoints, model downloads, and significant time.

Long training and evaluation are human-gated. Read [agent orchestration](agent_orchestration.md) and [reproducibility](reproducibility.md) before using any expensive command. Paths below are repository-relative; run them from the repository root.

## Historical environment setup

Python 3.12 was used locally because PyTorch CUDA wheels were more reliable there than on the system Python available during development. Package versions are not pinned, so this is an observed setup, not a reproducibility guarantee.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install numpy Pillow PyYAML tqdm tensorboard pytest
```

Latent SiT paths additionally used `diffusers`, `huggingface_hub`, and `safetensors`.

## Historical low-cost checks

```powershell
.\.venv\Scripts\python.exe -m pytest mini_diffusion\tests
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\cifar10_debug.yaml --overfit-smoke --overfit-updates 40
```

The debug CIFAR config historically used fake data by default. Even so, package availability and runtime behavior were not re-validated for this documentation update.

## CIFAR-10 DDPM commands

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\cifar10_debug.yaml
.\.venv\Scripts\python.exe mini_diffusion\train.py --config mini_diffusion\configs\cifar10_debug.yaml --resume outputs\cifar10_debug\checkpoints\latest.pt
.\.venv\Scripts\python.exe mini_diffusion\sample.py --checkpoint outputs\cifar10_debug\checkpoints\latest.pt --classes 0 1 2 3 --seeds 10 20 30 40 --guidance-scale 1.5 --output outputs\samples\cifar10_debug
```

Historical full training and benchmark commands are recorded in [CIFAR baseline](../reports/cifar10_baseline.md) and [optimisation report](../reports/cifar10_optimization_report.md). Treat them as expensive local operations, not setup steps.

## Tiny ImageNet commands

Tiny ImageNet requires a manually obtained archive and dataset validation. The original download, extraction, validation, debug, and full-run commands are retained in Git history and the project reports; the partial experiment was intentionally stopped. Consult [partial milestone](../reports/tiny_imagenet_partial.md) before attempting it.

## Latent SiT, Imagenette, and REPA

These paths require a pretrained VAE, Imagenette data, and—for REPA—cached DINOv2 features. Historical cache preparation, debug, sampling, and benchmark commands are recorded in [SiT evaluator setup](../reports/imagenette_sit_evaluator_setup.md) and [REPA setup](../reports/imagenette_sit_s_128_repa_setup.md). Do not infer that model downloads, paths, or providers are currently available.

## AFHQ Cats

AFHQ work requires the official data under a local ignored `datasets/afhq/` layout, a VAE, cached latents, and a CUDA-capable environment. Historical preparation, debug, benchmark, and training commands appear in [AFHQ setup](../reports/afhq_cat_sit_b_128_setup.md). The current selected model was reached through a human-gated early-stop REPA fork; its exact recorded training/evaluation command and limitations are in the [AFHQ result report](../reports/afhq_cat_sit_b_128_repa_early_stop_results.md).

No command in this guide should be read as permission to resume training, regenerate evaluation outputs, or modify the frozen checkpoint without a separately approved scope.


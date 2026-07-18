# Reproducibility and evidence guide

## What this repository supports today

The observed development environment was **Windows**, **Python 3.12**, and an **NVIDIA RTX 4090**. Some reports additionally record PyTorch/CUDA versions for their specific run. This is an observed local environment, not a portable support matrix: dependencies are not fully pinned, there is no lockfile or CI workflow, and GPU/model-download availability varies.

The repository tracks source code, configurations, compact reports, selected curated visuals, and append-only evidence ledgers. It intentionally ignores datasets, checkpoints, latent/feature caches, training logs, and full evaluation output. Where a local artifact mattered to a decision, its report records a repository-relative location and/or SHA-256.

## Cheap inspection path

Run these from the repository root in PowerShell. They are source/evidence inspections, not ML validation, and were **not re-run during portfolio packaging**.

```powershell
git status --short
git diff --check
Get-Content reports\afhq_cat_sit_b_128_repa_early_stop_results.md
Get-Content reports\portfolio_claim_evidence_matrix.md
```

For the historical test command, see [operations guide](operations_guide.md). It is not presented as a current portable verification path because package versions and environment setup remain unpinned.

## Training and evaluation boundaries

Training, sampling, cache preparation, benchmark, and evaluation commands may require a CUDA-capable environment, local datasets/checkpoints, downloaded model weights, and substantial time. They are expensive and human-gated by project policy; this guide does not authorize or imply running them. Historical commands are preserved in the [operations guide](operations_guide.md) and the relevant reports, with no claim that they were re-tested during packaging.

The current AFHQ Cats decision is based on a quick-200 protocol, not a full-1000 confirmation. The frozen checkpoint, dataset, and full evaluator artifacts are local/ignored; the tracked [result report](../reports/afhq_cat_sit_b_128_repa_early_stop_results.md) is the public evidence entry point.

## Before public release

The readiness audit identifies remaining release blockers: choose a code license, add consolidated third-party attribution/use policy, provide a genuinely portable pinned inexpensive verification path, and run a final secret/link/render review. [Audit](../reports/public_repo_readiness_audit.md)


# AFHQ Cats SiT-B/2 REPA Early-Stop Readiness

Date: 2026-07-18. Status: **prepared; training pending/not run**.

## Approved lineage

- Source: `outputs/afhq_cat_sit_b_128_repa/checkpoints/step_0010000.pt`
- Source step: `10000`
- Source SHA-256: `f7ee59598f3db71fffe75159bc51582a62925c25b3650d8ec514dcaae47d4068`
- New isolated output: `outputs/afhq_cat_sit_b_128_repa_early_stop/`
- Expected final raw checkpoint: `outputs/afhq_cat_sit_b_128_repa_early_stop/checkpoints/step_0020000.pt`
- Expected final resume checkpoint: `outputs/afhq_cat_sit_b_128_repa_early_stop/checkpoints/latest.pt`

The staged config is `mini_diffusion/configs/afhq_cat_sit_b_128_repa_early_stop.yaml`. It resumes the SiT raw weights, EMA, AdamW state for every SiT parameter, scheduler state, global step, and Python/NumPy/Torch/CUDA RNG states from the approved 10k REPA source. The source projector state and its optimizer segment are explicitly validated and then excluded from the new optimizer; the new run has no projector, DINO feature cache, feature loader, DINO teacher, or REPA-loss computation.

The fork refuses a source path, SHA-256, source step, latent-cache fingerprint, model/optimizer/scheduler/EMA recipe, or output directory that differs from the approved lineage. It also refuses any non-empty fork output directory, so it cannot overwrite or mix a lineage with the always-on REPA run.

## Manual training command

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128_repa_early_stop.yaml --resume outputs\afhq_cat_sit_b_128_repa\checkpoints\step_0010000.pt --max-steps 20000
```

This is the only approved long-run command for this stage. It has **not** been run by this preparation task. `--max-steps 20000` is only the stop-at limit; the restored scheduler remains on its 100,000-step horizon.

## Runtime stop conditions

Stop immediately and preserve the console output if any of the following occurs:

- source SHA-256, global step, latent cache fingerprint, architecture, optimizer/scheduler/EMA recipe, or fork output safety validation fails;
- a non-finite loss, parameter, gradient, EMA value, or decoded preview appears;
- a checkpoint write fails, a milestone already exists, or the run tries to use the always-on REPA output directory;
- startup reports a nonzero projector parameter count, REPA metric tags, a DINO/feature-cache load, or a nonzero REPA coefficient;
- CUDA OOM or a persistent runtime/data-loader error occurs.

On normal completion, bring back the terminal log, SHA-256 plus `global_step` for `step_0020000.pt` and `latest.pt`, the new TensorBoard/log location, fixed raw/EMA previews at 15k and 20k, and the Git status. Do not run sampling, evaluation, benchmark, or full-1000 protocol yet. The supervisor will then specify the unified quick-200 comparison across baseline raw 20k, always-on REPA raw 20k, and early-stop raw 20k, with EMA diagnostic-only.

## Preparation checks actually run

```powershell
.\.venv\Scripts\python.exe -m pytest mini_diffusion\tests\test_sit.py mini_diffusion\tests\test_evaluate_comparison.py -q
```

Result: `15 passed in 2.49s`. This CPU-only targeted unit suite covered normal SiT checkpoint/scheduler resume, the explicit REPA-to-no-REPA optimizer-state translation, preservation of SiT/EMA/scheduler state, refusal of a non-empty fork output, and rendering of the numeric `repa_raw_vs_ema_20k` report section. `python -m py_compile mini_diffusion\train_sit.py mini_diffusion\evaluate_comparison.py` also passed. No training, sampling, evaluation, CUDA benchmarking, or cache preparation was run.

A read-only inspection of the actual source checkpoint also passed: its SHA-256 matches the config, `global_step=10000`, the stored source recipe matches the staged SiT/optimizer/scheduler/EMA recipe, it contains a REPA projector, and its one AdamW group contains 137 parameters (the SiT segment plus the explicitly discarded projector segment). This inspection did not modify the source checkpoint or create output artifacts.

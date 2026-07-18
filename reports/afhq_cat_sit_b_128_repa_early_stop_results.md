# AFHQ Cats SiT-B/2 REPA Early-Stop: 20k Results

Date: 2026-07-18. Scope: approved `REPA 10k -> REPA OFF -> 20k` fork and unified raw quick-200 comparison.

## Training milestone verified from artifacts

The long run was human-gated and user-reported. The approved command was:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\train_sit.py --config mini_diffusion\configs\afhq_cat_sit_b_128_repa_early_stop.yaml --resume outputs\afhq_cat_sit_b_128_repa\checkpoints\step_0010000.pt --max-steps 20000
```

Read-only checkpoint inspection confirmed the isolated output has `step_0015000.pt`, `step_0020000.pt`, `latest.pt`, raw/EMA previews at 15k and 20k, and a TensorBoard event file. The source stated in the approved readiness record is `outputs/afhq_cat_sit_b_128_repa/checkpoints/step_0010000.pt` (SHA-256 `f7ee59598f3db71fffe75159bc51582a62925c25b3650d8ec514dcaae47d4068`). The forked 20k checkpoint records `global_step=20000`, cache fingerprint `c94251a82755ce2c1e02269404b3d5b92b5f2325fcefe51e53694c316b4e5612`, no `repa` or `repa_projector` state, and finite raw/EMA tensors.

| Artifact | SHA-256 | Verified state |
| --- | --- | --- |
| `outputs/afhq_cat_sit_b_128_repa_early_stop/checkpoints/step_0015000.pt` | `fa177609aa3715a32eb206ea48523d49c7c91b2c9583d9197799433dcd22e963` | global step 15000, finite |
| `outputs/afhq_cat_sit_b_128_repa_early_stop/checkpoints/step_0020000.pt` | `300b5600b86d1a35ebf2c27307e480070cceee113735b23ffca8e46316e57bd0` | global step 20000, finite |
| `outputs/afhq_cat_sit_b_128_repa_early_stop/checkpoints/latest.pt` | `c859c448635d490946ec401bad8050b8c594179a90b7f6c0cdacee29052866d2` | global step 20000, finite |

No exact duration, GPU telemetry, or retained console log was available, so those fields are explicitly unknown rather than inferred.

## Unified raw quick-200

Command actually run:

```powershell
.\.venv\Scripts\python.exe mini_diffusion\evaluate_comparison.py --config mini_diffusion\configs\evaluation\afhq_cat_baseline_repa_early_stop_20k.yaml
```

Protocol: official held-out AFHQ Cats test split, 200 fixed seeds (`1000-1199`), class 0, Heun-50, CFG 1.0, shared VAE/Inception features. The evaluator actually completed twice: canonical event `afhq-cats-baseline-vs-repa-quick-10k-20k-20260718-3` at `2026-07-18T14:46:11.047434Z` (69.637512 s), then redundant verification event `afhq-cats-baseline-vs-repa-quick-10k-20k-20260718-4` at `2026-07-18T14:47:03.475555Z` (69.857339 s). The second run used the same seeds and protocol; it is not an independent statistical replicate. Its complete stored metric and change values are identical to the canonical event, and both runs recorded unchanged hashes for all three checkpoints.

| Raw 20k variant | FID | KID | Precision | Recall | Finite / black-white failures | Duplicate pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 48.051 | 0.02052 | 0.340 | 0.754 | 0 / 0 | 0 |
| Always-on REPA | 52.384 | 0.02531 | 0.310 | 0.722 | 0 / 0 | 0 |
| Early-stop REPA | 45.787 | 0.01692 | 0.280 | 0.732 | 0 / 0 | 0 |

Early-stop versus baseline: FID `-2.264` (`-4.71%`), KID `-0.003596` (`-17.53%`), precision `-0.060` (`-17.65%`), recall `-0.022` (`-2.92%`). Early-stop versus always-on REPA: FID `-6.597` (`-12.59%`), KID `-0.008384` (`-33.13%`), precision `-0.030` (`-9.68%`), recall `+0.010` (`+1.39%`). All checkpoints were hash-checked before and after each evaluation; fixed-seed one-image probes were bitwise deterministic.

Artifacts: `evaluation/afhq_cat_baseline_repa_early_stop_20k/comparison.json`, `metrics.csv`, `report.md`, `grids/`, `comparisons/`, and `nearest_neighbors/`.

## Evidence-limited conclusion

Early-stop clearly beats always-on REPA on FID/KID and also has a lower FID/KID than baseline, so switching REPA off at 10k produced useful evidence. It does not dominate baseline: baseline remains better on precision and recall. The quick-200 result identifies early-stop as a plausible finalist, but the trade-off is not unambiguously sufficient by itself to select or freeze a model. Supervisor review is required before any full-1000 run.

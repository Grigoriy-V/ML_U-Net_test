import pytest

from mini_diffusion.evaluate_comparison import delta, failure_count, finite_state, reference_variant_id, write_markdown


def test_comparison_delta_and_failures() -> None:
    left = {"metrics": {"fid": 10.0, "kid": 0.2, "precision": 0.4, "recall": 0.5}}
    right = {"metrics": {"fid": 8.0, "kid": 0.1, "precision": 0.5, "recall": 0.4}}
    changes = delta(left, right)
    assert changes["fid"] == {"absolute": -2.0, "percent": -20.0}
    metrics = {"pixel": {"finite_failures": 2, "black_white_failures": 3}}
    assert failure_count(metrics) == 5


def test_finite_state_rejects_nonfinite_tensors() -> None:
    import torch

    assert finite_state({"weight": torch.ones(2)})
    assert not finite_state({"weight": torch.tensor([float("nan")])})


def test_reference_variant_can_override_legacy_baseline_10k_default() -> None:
    inspected = {"baseline_raw_20k": object()}
    assert reference_variant_id({"reference_variant": "baseline_raw_20k"}, inspected) == "baseline_raw_20k"
    with pytest.raises(ValueError, match="Missing reference variant"):
        reference_variant_id({}, inspected)


def test_report_includes_raw_vs_ema_numeric_delta(tmp_path) -> None:
    changes = delta(
        {"metrics": {"fid": 10.0, "kid": 0.2, "precision": 0.4, "recall": 0.5}},
        {"metrics": {"fid": 15.0, "kid": 0.3, "precision": 0.2, "recall": 0.6}},
    )
    report = tmp_path / "report.md"
    write_markdown({"rows": [], "changes": {"repa_raw_vs_ema_20k": changes}, "checkpoint_unchanged": True, "deterministic_sampling": True}, report)
    text = report.read_text(encoding="utf-8")
    assert "### repa_raw_vs_ema_20k" in text
    assert "- fid: +5.000000 (+50.00%)" in text

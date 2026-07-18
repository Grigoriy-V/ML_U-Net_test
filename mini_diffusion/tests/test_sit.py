from __future__ import annotations

import copy

import torch

from mini_diffusion.diffusion import EMA
from mini_diffusion.latent_cache import CACHE_FORMAT_VERSION, load_cache, validate_cache
from mini_diffusion.sit import SiT, linear_interpolant, sample_ode, velocity_loss
from mini_diffusion.train_sit import build_model, build_optimizer, build_scheduler, mean_step_metrics, resume_checkpoint, resume_repa_off_fork, save_checkpoint, training_limits, validate_repa_off_fork_target


def tiny_sit() -> SiT:
    return SiT(input_size=16, hidden_size=48, depth=2, num_heads=6, mlp_ratio=2.0, cond_drop_prob=0.0)


def test_cache_round_trip_and_validation(tmp_path) -> None:
    payload = {"latents": torch.randn(3, 4, 16, 16).half(), "labels": torch.tensor([0, 1, 9]), "relative_paths": ["a.jpg", "b.jpg", "c.jpg"], "metadata": {"format_version": CACHE_FORMAT_VERSION, "dataset": "imagenette-160", "split": "train", "resolution": 128, "vae_model_id": "stabilityai/sd-vae-ft-mse", "vae_revision": None, "latent_scaling_factor": 0.18215, "cache_seed": 123, "preprocessing": "test"}}
    path = tmp_path / "cache.pt"; torch.save(payload, path)
    assert load_cache(path, expected_resolution=128, expected_vae_model_id="stabilityai/sd-vae-ft-mse")["latents"].shape == (3, 4, 16, 16)
    broken = copy.deepcopy(payload); broken["metadata"].pop("cache_seed")
    try: validate_cache(broken)
    except ValueError: pass
    else: raise AssertionError("invalid metadata was accepted")


def test_patchify_and_forward_labels_and_null() -> None:
    model = tiny_sit(); x = torch.randn(2, 4, 16, 16); t = torch.tensor([0.2, 0.8])
    assert model(x, t, torch.tensor([1, 2])).shape == x.shape
    assert model(x, t, torch.full((2,), model.null_class)).shape == x.shape


def test_interpolant_endpoints_and_finite_backward() -> None:
    x0, noise = torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16)
    xt0, target = linear_interpolant(x0, noise, torch.zeros(2)); xt1, _ = linear_interpolant(x0, noise, torch.ones(2))
    assert torch.equal(xt0, x0) and torch.equal(xt1, noise) and target.shape == x0.shape
    model = tiny_sit(); loss = velocity_loss(model(xt0, torch.zeros(2), torch.tensor([0, 1])), target); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_euler_heun_cfg_and_determinism() -> None:
    model = tiny_sit().eval(); labels = torch.tensor([1])
    for sampler in ("euler", "heun"):
        first = sample_ode(model, (1, 4, 16, 16), labels, torch.device("cpu"), steps=3, sampler=sampler, guidance_scale=1.5, generator=torch.Generator().manual_seed(7), diagnostics=True)
        second = sample_ode(model, (1, 4, 16, 16), labels, torch.device("cpu"), steps=3, sampler=sampler, guidance_scale=1.5, generator=torch.Generator().manual_seed(7), diagnostics=True)
        assert first.shape == (1, 4, 16, 16) and torch.isfinite(first).all() and torch.equal(first, second)


def test_sit_checkpoint_roundtrip_and_resume(tmp_path) -> None:
    cfg = {"data": {"latent_resolution": 16, "num_classes": 10}, "model": {"patch_size": 2, "hidden_size": 48, "depth": 2, "num_heads": 6, "mlp_ratio": 2.0, "cond_drop_prob": 0.0}, "train": {"learning_rate": 1e-3, "weight_decay": 0.0, "ema_decay": 0.9}}
    model = build_model(cfg); optimizer = build_optimizer(model, cfg); ema = EMA(model, 0.9); path = tmp_path / "latest.pt"; save_checkpoint(path, model, optimizer, ema, cfg, 5, "test")
    restored = build_model(cfg); restored_optimizer = build_optimizer(restored, cfg); restored_ema = EMA(restored, 0.9)
    assert resume_checkpoint(str(path), restored, restored_optimizer, restored_ema, torch.device("cpu")) == 5
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))


def test_sit_scheduler_roundtrip_and_resume(tmp_path) -> None:
    cfg = {"data": {"latent_resolution": 16, "num_classes": 1}, "model": {"patch_size": 2, "hidden_size": 48, "depth": 2, "num_heads": 6, "mlp_ratio": 2.0, "cond_drop_prob": 0.0}, "train": {"learning_rate": 1e-3, "min_learning_rate": 1e-4, "warmup_steps": 3, "weight_decay": 0.0, "ema_decay": 0.9}}
    model = build_model(cfg); optimizer = build_optimizer(model, cfg); scheduler = build_scheduler(optimizer, cfg, 20); ema = EMA(model, 0.9)
    for _ in range(5): optimizer.step(); scheduler.step()
    path = tmp_path / "scheduled.pt"; save_checkpoint(path, model, optimizer, ema, cfg, 5, "test", scheduler=scheduler)
    restored = build_model(cfg); restored_optimizer = build_optimizer(restored, cfg); restored_scheduler = build_scheduler(restored_optimizer, cfg, 20); restored_ema = EMA(restored, 0.9)
    assert resume_checkpoint(str(path), restored, restored_optimizer, restored_ema, torch.device("cpu"), scheduler=restored_scheduler) == 5
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_repa_off_fork_preserves_sit_optimizer_scheduler_ema_and_rng(tmp_path) -> None:
    import hashlib
    from torch import nn

    source_cfg = {"data": {"resolution": 128, "latent_resolution": 16, "num_classes": 1}, "vae": {"model_id": "stabilityai/sd-vae-ft-mse", "revision": None}, "model": {"patch_size": 2, "hidden_size": 48, "depth": 2, "num_heads": 6, "mlp_ratio": 2.0, "cond_drop_prob": 0.0}, "repa": {"enabled": True}, "train": {"learning_rate": 1e-3, "min_learning_rate": 1e-4, "warmup_steps": 3, "weight_decay": 0.0, "grad_accum_steps": 1, "grad_clip": 1.0, "ema_decay": 0.9, "scheduler_total_steps": 20}}
    source_model, projector = build_model(source_cfg), nn.Linear(48, 8)
    source_modules = nn.ModuleList([source_model, projector]); source_optimizer = build_optimizer(source_modules, source_cfg); source_scheduler = build_scheduler(source_optimizer, source_cfg, 20); source_ema = EMA(source_model, 0.9)
    source_optimizer.zero_grad()
    sum(parameter.square().mean() for parameter in source_modules.parameters()).backward()
    source_optimizer.step(); source_scheduler.step(); source_ema.update(source_model)
    source_path = tmp_path / "repa_10k.pt"; save_checkpoint(source_path, source_model, source_optimizer, source_ema, source_cfg, 10000, "cache", projector=projector, feature_metadata={"fingerprint": "features"}, scheduler=source_scheduler)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    target_cfg = {**source_cfg, "repa": {"enabled": False}, "fork_repa_off": {"source_checkpoint": str(source_path), "source_step": 10000, "source_sha256": digest}}
    target_model = build_model(target_cfg); target_optimizer = build_optimizer(target_model, target_cfg); target_scheduler = build_scheduler(target_optimizer, target_cfg, 20); target_ema = EMA(target_model, 0.9)
    assert resume_repa_off_fork(str(source_path), target_model, target_optimizer, target_ema, torch.device("cpu"), target_cfg, "cache", target_scheduler) == 10000
    assert all(torch.equal(left, right) for left, right in zip(source_model.parameters(), target_model.parameters()))
    assert target_scheduler.state_dict() == source_scheduler.state_dict()
    assert len(target_optimizer.state_dict()["param_groups"][0]["params"]) == len(list(target_model.parameters()))
    assert all(torch.equal(source_ema.shadow[name], target_ema.shadow[name]) for name in source_ema.shadow)


def test_repa_off_fork_refuses_nonempty_target_output(tmp_path) -> None:
    output = tmp_path / "fork-output"; output.mkdir(); (output / "unexpected.txt").write_text("do not mix lineages", encoding="utf-8")
    cfg = {"output_dir": str(output), "repa": {"enabled": False}, "fork_repa_off": {"source_checkpoint": str(tmp_path / "source.pt"), "source_step": 10000, "source_sha256": "unused"}}
    try:
        validate_repa_off_fork_target(cfg, str(tmp_path / "source.pt"))
    except FileExistsError: pass
    else: raise AssertionError("non-empty fork output was accepted")


def test_cli_stop_step_does_not_shorten_scheduler_horizon() -> None:
    cfg = {"train": {"max_steps": 100000, "scheduler_total_steps": 100000}}
    assert training_limits(cfg, 10000) == (10000, 100000)


def test_grad_accumulation_logging_uses_microbatch_means() -> None:
    metrics = mean_step_metrics({"loss": 3.0, "flow_loss": 2.0, "repa_loss": 1.0, "repa_weighted_loss": 0.5}, 2)
    assert metrics == {"loss": 1.5, "flow_loss": 1.0, "repa_loss": 0.5, "repa_weighted_loss": 0.25}


def test_foreach_ema_updates() -> None:
    model = tiny_sit()
    ema = EMA(model, decay=0.5, foreach=True)
    before = {name: value.clone() for name, value in ema.shadow.items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    assert any(not torch.equal(before[name], value) for name, value in ema.shadow.items())


def test_one_batch_overfit_velocity() -> None:
    torch.manual_seed(0); model = SiT(input_size=16, hidden_size=48, depth=2, num_heads=6, mlp_ratio=2.0, cond_drop_prob=0.0); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x0, noise, labels, t = torch.randn(4, 4, 16, 16), torch.randn(4, 4, 16, 16), torch.tensor([0, 1, 2, 3]), torch.tensor([0.2, 0.4, 0.6, 0.8]); xt, target = linear_interpolant(x0, noise, t); losses = []
    for _ in range(40):
        optimizer.zero_grad(); loss = velocity_loss(model(xt, t, labels), target); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10 * 0.75

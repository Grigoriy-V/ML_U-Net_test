from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from urllib.request import urlopen
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mini_diffusion.data import AFHQCatDataset, build_imagenette
from mini_diffusion.latent_cache import cache_fingerprint, load_cache
from mini_diffusion.repa import (
    REPA_CACHE_FORMAT_VERSION,
    atomic_replace_directory,
    feature_cache_fingerprint,
    feature_cache_paths,
    new_staging_directory,
    pool_teacher_grid,
)
from mini_diffusion.train_sit import load_config


DINO_SOURCE = "facebookresearch/dinov2"
DINO_MODEL = "dinov2_vitb14"
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def load_dino(device: torch.device, teacher_name: str) -> tuple[torch.nn.Module, str]:
    if teacher_name != DINO_MODEL:
        raise ValueError(f"Only the configured reference teacher {DINO_MODEL!r} is supported, got {teacher_name!r}")
    model = torch.hub.load(DINO_SOURCE, teacher_name, trust_repo=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    try:
        revision = subprocess.check_output(["git", "-C", str(hub_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        try:
            with urlopen("https://api.github.com/repos/facebookresearch/dinov2/commits/main", timeout=10) as response:
                revision = json.load(response)["sha"]
        except (OSError, KeyError, json.JSONDecodeError):
            revision = "unavailable"
    return model, revision


def dino_input(images: torch.Tensor) -> torch.Tensor:
    # Images are the exact deterministic 128px VAE-cache crop, normalized to [-1, 1].
    rgb = F.interpolate(((images.float() + 1.0) * 0.5).clamp(0, 1), size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor(DINO_MEAN, device=rgb.device)[None, :, None, None]
    std = torch.tensor(DINO_STD, device=rgb.device)[None, :, None, None]
    return (rgb - mean) / std


def source_indices(dataset, relative_paths: list[str]) -> list[int]:
    root = Path(dataset.root).resolve()
    lookup = {str(Path(path).resolve().relative_to(root)): index for index, (path, _) in enumerate(dataset.samples)}
    missing = [path for path in relative_paths if path not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} latent-cache paths are absent from Imagenette; first: {missing[0]}")
    return [lookup[path] for path in relative_paths]


def source_dataset(cfg: dict, latent: dict, split: str, download_dataset: bool):
    data_cfg = cfg["data"]
    if data_cfg["dataset"] == "imagenette-160":
        dataset = build_imagenette(data_cfg["root"], split, data_cfg["resolution"], download=download_dataset)
        indices = source_indices(dataset, latent["relative_paths"])
        expected_labels = latent["labels"].tolist()
        actual_labels = [dataset.samples[index][1] for index in indices]
        if actual_labels != expected_labels:
            raise ValueError("Imagenette labels do not match latent cache labels for relative paths")
        return Subset(dataset, indices), "none"
    if data_cfg["dataset"] == "afhq-cat-official":
        if split != "train":
            raise ValueError("AFHQ REPA features are only required for the train cache")
        dataset = AFHQCatDataset(
            data_cfg["root"], "train", int(data_cfg["resolution"]),
            augmentation_variants=int(data_cfg["augmentation_variants"]),
            seed=int(data_cfg.get("augmentation_seed", cfg.get("seed", 123))),
            crop_scale=tuple(float(value) for value in data_cfg["augmentation_scale"]),
        )
        manifest = dataset.manifest()[:len(latent["relative_paths"])]
        if [entry["source_path"] for entry in manifest] != latent["relative_paths"]:
            raise ValueError("AFHQ feature preprocessing order does not match the latent cache manifest")
        manifest_path = Path(data_cfg["cache_dir"]) / "train_manifest.jsonl"
        cached_manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()][:len(manifest)]
        expected_manifest = [
            {key: entry[key] for key in ("source_path", "augmentation_seed", "augmentation_index", "split", "sha256")}
            for entry in manifest
        ]
        actual_manifest = [
            {key: entry[key] for key in ("source_path", "augmentation_seed", "augmentation_index", "split", "sha256")}
            for entry in cached_manifest
        ]
        if actual_manifest != expected_manifest:
            raise ValueError("AFHQ feature preprocessing manifest does not match the latent cache manifest")
        if len(manifest) != len(latent["labels"]) or any(label != 0 for label in latent["labels"].tolist()):
            raise ValueError("AFHQ feature labels do not match the latent cache")
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        return Subset(dataset, range(len(manifest))), f"deterministic RandomResizedCrop(scale=0.85-1.0, ratio=1.0) + horizontal flip; exact AFHQ train manifest order; manifest_sha256={digest}"
    raise ValueError(f"REPA feature preparation does not support dataset={data_cfg['dataset']!r}")


def prepare_split(cfg: dict, split: str, limit: int | None, force: bool, download_dataset: bool) -> Path:
    repa_cfg, data_cfg = cfg["repa"], cfg["data"]
    if int(repa_cfg["teacher_input_resolution"]) != 224 or int(repa_cfg["teacher_feature_dim"]) != 768:
        raise ValueError("dinov2_vitb14 REPA requires teacher_input_resolution=224 and teacher_feature_dim=768")
    latent_path = Path(data_cfg["cache_dir"]) / f"{split}.pt"
    latent = load_cache(latent_path, expected_resolution=data_cfg["resolution"], expected_vae_model_id=cfg["vae"]["model_id"])
    if limit is not None:
        latent = {**latent, "latents": latent["latents"][:limit], "labels": latent["labels"][:limit], "relative_paths": latent["relative_paths"][:limit]}
    ordered, augmentation = source_dataset(cfg, latent, split, download_dataset)
    destination = Path(repa_cfg["feature_cache_dir"]) / split
    if destination.exists() and not force:
        raise FileExistsError(f"Feature cache exists: {destination}. Pass --force to overwrite it.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher, revision = load_dino(device, repa_cfg["teacher"])
    batch_size = int(data_cfg.get("prepare_batch_size", 16))
    workers = int(data_cfg.get("num_workers", 0))
    loader_kwargs = {"batch_size": batch_size, "shuffle": False, "num_workers": workers, "pin_memory": device.type == "cuda"}
    if workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": int(data_cfg.get("prefetch_factor", 2))})
    loader = DataLoader(ordered, **loader_kwargs)
    staging = new_staging_directory(destination)
    paths = feature_cache_paths(staging)
    features = np.lib.format.open_memmap(paths["features"], mode="w+", dtype=np.float16, shape=(len(ordered), 64, 768))
    labels = np.lib.format.open_memmap(paths["labels"], mode="w+", dtype=np.int64, shape=(len(ordered),))
    offset = 0
    with torch.inference_mode():
        for images, batch_labels in tqdm(loader, desc=f"dino_{split}"):
            result = teacher.forward_features(dino_input(images.to(device, non_blocking=True)))
            tokens = result["x_norm_patchtokens"] if isinstance(result, dict) else result
            pooled = pool_teacher_grid(tokens).cpu().to(torch.float16).numpy()
            count = pooled.shape[0]
            features[offset:offset + count] = pooled
            labels[offset:offset + count] = batch_labels.numpy()
            offset += count
    features.flush(); labels.flush()
    feature_view = np.load(paths["features"], mmap_mode="r")
    metadata = {
        "format_version": REPA_CACHE_FORMAT_VERSION, "teacher": repa_cfg["teacher"], "teacher_source": DINO_SOURCE,
        "teacher_revision": revision, "teacher_input_resolution": int(repa_cfg["teacher_input_resolution"]),
        "preprocessing": "VAE-cache deterministic RGB128 crop -> Resize(224,bicubic,antialias) -> ImageNet normalize",
        "latent_cache_fingerprint": cache_fingerprint(latent), "split": split, "feature_shape": list(feature_view.shape),
        "feature_dtype": str(feature_view.dtype), "pooling": "adaptive_avg_pool2d 16x16 -> 8x8; row-major [B,64,768]",
        "class_mapping": sorted(set(int(x) for x in labels.tolist())), "augmentation": augmentation,
        "statistics": {"mean": float(feature_view.mean(dtype=np.float64)), "std": float(feature_view.std(dtype=np.float64)), "min": float(feature_view.min()), "max": float(feature_view.max()), "finite": bool(np.isfinite(feature_view).all()), "size_bytes": int(paths["features"].stat().st_size)},
    }
    metadata["fingerprint"] = feature_cache_fingerprint(metadata, paths["features"])
    paths["paths"].write_text(json.dumps(latent["relative_paths"], ensure_ascii=True), encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    # Windows refuses to rename a directory while numpy mmap views still hold file handles.
    del features, labels, feature_view
    gc.collect()
    atomic_replace_directory(staging, destination, force)
    print("feature_cache_statistics: " + json.dumps(metadata, sort_keys=True))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Create frozen DINOv2 features for cached Imagenette VAE latents.")
    parser.add_argument("--config", required=True); parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--limit", type=int); parser.add_argument("--force", action="store_true"); parser.add_argument("--download-dataset", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    output = prepare_split(load_config(args.config), args.split, args.limit, args.force, args.download_dataset)
    print(f"feature_cache_written: {output}")


if __name__ == "__main__":
    main()

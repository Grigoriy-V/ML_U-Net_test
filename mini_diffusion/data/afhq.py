from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


AFHQ_OFFICIAL_REPOSITORY = "https://github.com/clovaai/stargan-v2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def afhq_root(root: str | Path) -> Path:
    root = Path(root)
    return root / "afhq" if (root / "afhq").is_dir() else root


def cat_split_dir(root: str | Path, split: str) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("AFHQ cat split must be 'train' or 'test'")
    dataset_root = afhq_root(root)
    candidates = [dataset_root / split / "cat"]
    # StarGAN v2 calls the official held-out set `val`; retain it as a
    # read-only test alias, never a training source.
    if split == "test":
        candidates.append(dataset_root / "val" / "cat")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    expected = dataset_root / split / "cat"
    raise FileNotFoundError(
        f"AFHQ cat {split} split was not found at {expected}. Download the official AFHQ archive from "
        f"{AFHQ_OFFICIAL_REPOSITORY} and extract either train/cat + test/cat or train/cat + val/cat under {dataset_root}."
    )


def cat_files(root: str | Path, split: str) -> list[Path]:
    directory = cat_split_dir(root, split)
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"AFHQ cat {split} split contains no supported images: {directory}")
    return files


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _square_crop(image: Image.Image, seed: int | None, train: bool) -> Image.Image:
    width, height = image.size
    if train:
        rng = random.Random(seed)
        side = max(1, int(min(width, height) * rng.uniform(0.80, 1.00)))
        left = rng.randint(0, max(0, width - side))
        top = rng.randint(0, max(0, height - side))
        image = image.crop((left, top, left + side, top + side))
        if rng.random() < 0.5:
            image = TF.hflip(image)
    else:
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    return image


def preprocess_cat(image: Image.Image, resolution: int, *, seed: int | None, train: bool):
    image = _square_crop(image.convert("RGB"), seed, train)
    image = TF.resize(image, [resolution, resolution], interpolation=InterpolationMode.BICUBIC, antialias=True)
    return TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3)


@dataclass(frozen=True)
class AFHQCatEntry:
    path: Path
    split: str
    augmentation_seed: int


class AFHQCatDataset(Dataset):
    def __init__(self, root: str | Path, split: str, resolution: int, *, augmentation_variants: int = 1, seed: int = 123) -> None:
        if split == "train" and augmentation_variants < 1:
            raise ValueError("augmentation_variants must be positive")
        self.root = afhq_root(root).resolve()
        files = cat_files(self.root, split)
        variants = augmentation_variants if split == "train" else 1
        self.entries = [
            AFHQCatEntry(path=path, split=split, augmentation_seed=seed + file_index * variants + variant)
            for file_index, path in enumerate(files) for variant in range(variants)
        ]
        self.resolution, self.train = resolution, split == "train"

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        with Image.open(entry.path) as image:
            pixels = preprocess_cat(image, self.resolution, seed=entry.augmentation_seed, train=self.train)
        return pixels, 0

    def manifest(self) -> list[dict[str, str | int]]:
        return [
            {
                "source_path": str(entry.path.relative_to(self.root)).replace("\\", "/"),
                "augmentation_seed": entry.augmentation_seed,
                "split": entry.split,
                "sha256": sha256_file(entry.path),
            }
            for entry in self.entries
        ]

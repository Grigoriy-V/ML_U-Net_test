from __future__ import annotations

import hashlib

from PIL import Image
import torch

from mini_diffusion.data import AFHQCatDataset, cat_split_dir


def write_image(path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 40))
    image.putdata([((color[0] + x * 7) % 256, (color[1] + y * 9) % 256, (color[2] + x * 3 + y * 5) % 256) for y in range(40) for x in range(48)])
    image.save(path)


def test_official_afhq_train_test_split_and_manifest(tmp_path) -> None:
    root = tmp_path / "afhq"
    write_image(root / "train" / "cat" / "train.png", (20, 30, 40))
    write_image(root / "test" / "cat" / "test.png", (50, 60, 70))
    dataset = AFHQCatDataset(root, "train", 16, augmentation_variants=4, seed=9)
    assert len(dataset) == 4 and cat_split_dir(root, "test").name == "cat"
    first, label = dataset[0]
    assert label == 0 and first.shape == (3, 16, 16) and torch.equal(first, dataset[0][0])
    manifest = dataset.manifest()
    assert {entry["split"] for entry in manifest} == {"train"}
    assert len({entry["augmentation_seed"] for entry in manifest}) == 4
    assert all(entry["source_path"].startswith("train/cat/") and len(entry["sha256"]) == 64 for entry in manifest)
    variant_hashes = {hashlib.sha256(dataset[index][0].numpy().tobytes()).hexdigest() for index in range(len(dataset))}
    assert len(variant_hashes) == 4


def test_official_afhq_val_is_test_alias(tmp_path) -> None:
    root = tmp_path / "afhq"; write_image(root / "train" / "cat" / "train.png", (1, 2, 3)); write_image(root / "val" / "cat" / "heldout.png", (4, 5, 6))
    test_dataset = AFHQCatDataset(root, "test", 16, augmentation_variants=4, seed=1)
    assert len(test_dataset) == 1 and test_dataset.entries[0].path.name == "heldout.png"

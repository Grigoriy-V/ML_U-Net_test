from __future__ import annotations

from PIL import Image
import torch

from mini_diffusion.data import AFHQCatDataset, cat_split_dir


def write_image(path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 20), color).save(path)


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


def test_official_afhq_val_is_test_alias(tmp_path) -> None:
    root = tmp_path / "afhq"; write_image(root / "train" / "cat" / "train.png", (1, 2, 3)); write_image(root / "val" / "cat" / "heldout.png", (4, 5, 6))
    test_dataset = AFHQCatDataset(root, "test", 16, augmentation_variants=4, seed=1)
    assert len(test_dataset) == 1 and test_dataset.entries[0].path.name == "heldout.png"

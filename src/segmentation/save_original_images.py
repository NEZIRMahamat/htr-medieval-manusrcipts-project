from __future__ import annotations

"""
Save original CATMuS segmentation page images to disk.

"""

import os

from datasets import Image as HFImage

try:
    from src.dataset import outputs_dir
    from src.utils import get_image_file_name, load_segmentation_dataset, select_split_subset
except ImportError:
    from dataset import outputs_dir
    from utils import get_image_file_name, load_segmentation_dataset, select_split_subset


def save_original_images(
    dataset,
    split: str = "train",
    save_dir: str | None = None,
    max_examples: int | None = 5,
    unique_names: bool = True,
) -> list[str]:
    """
    Save the original (unresized, unmodified) page images of a CATMuS
    segmentation split to disk.
    """
    save_dir = save_dir or os.path.join(outputs_dir, "segmentation", "images", split)
    os.makedirs(save_dir, exist_ok=True)

    raw_dataset = dataset.cast_column("image", HFImage(decode=False))
    split_dataset = select_split_subset(dataset[split], max_examples=max_examples)
    raw_split_dataset = select_split_subset(raw_dataset[split], max_examples=max_examples)

    saved_paths = []
    for i, example in enumerate(split_dataset):
        raw_example = raw_split_dataset[i]
        image_file_name = get_image_file_name(
            raw_example["image"],
            fallback_name=f"{split}_{i:06d}.png",
        )

        if unique_names:
            image_file_name = f"{split}_{i:06d}_{image_file_name}"

        image_path = os.path.join(save_dir, image_file_name)
        example["image"].convert("RGB").save(image_path)
        saved_paths.append(image_path)

    return saved_paths


def main_save_original_images() -> None:
    """
    Save the original images with the same split/count used to generate the
    existing train masks (outputs/segmentation/masks/train), so each saved
    image lines up 1:1 with its mask by file stem.
    """
    dataset = load_segmentation_dataset()

    for split in ["train"]:
        saved_paths = save_original_images(
            dataset,
            split=split,
            max_examples=5,
            unique_names=True,
        )
        print(f"Saved {len(saved_paths)} original images for split '{split}':")
        for path in saved_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main_save_original_images()

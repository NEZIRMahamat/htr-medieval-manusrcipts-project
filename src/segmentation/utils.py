from __future__ import annotations

"""
Utilities for CATMuS line-mask generation and U-Net preprocessing.
"""

import os
from collections.abc import Iterator

import numpy as np
from datasets import Image as HFImage
from matplotlib import pyplot as plt
from PIL import Image as PILImage, ImageDraw
from scipy.ndimage import label

try:
    from src.segmentation.dataset import data_dir, outputs_dir, load_data_from_dir
except ImportError:
    from dataset import data_dir, outputs_dir, load_data_from_dir



DEFAULT_IMAGE_SIZE = (256, 256)  # (height, width)


def normalize_target_size(target_size: tuple[int, int]) -> tuple[int, int]:
    """
    Validate and normalize a target image size expressed as (height, width).
    """
    if len(target_size) != 2:
        raise ValueError("target_size must be a tuple: (height, width).")

    height, width = int(target_size[0]), int(target_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("target_size dimensions must be positive.")

    return height, width


def target_size_to_pil_size(target_size: tuple[int, int]) -> tuple[int, int]:
    """
    Convert ML shape convention (height, width) to PIL convention (width, height).
    """
    height, width = normalize_target_size(target_size)
    return width, height


def get_image_file_name(raw_image: dict, fallback_name: str) -> str:
    """
    Extract the original image file name from a HuggingFace Image(decode=False).
    """
    image_path = raw_image.get("path") or fallback_name
    return os.path.basename(image_path)


def get_mask_file_name(image_file_name: str) -> str:
    """
    Build the mask file name from the original image file name.
    """
    image_stem, _ = os.path.splitext(image_file_name)
    return f"{image_stem}_mask.png"


def save_masks_to_disk(mask: np.ndarray, save_dir: str, mask_file_name: str) -> str:
    """
    Save a binary mask to disk as a PNG image.
    """
    os.makedirs(save_dir, exist_ok=True)

    if not mask_file_name.lower().endswith(".png"):
        mask_file_name = f"{mask_file_name}.png"

    mask_path = os.path.join(save_dir, mask_file_name)
    mask_array = np.squeeze(mask)
    mask_image = PILImage.fromarray((mask_array > 0).astype(np.uint8) * 255)
    mask_image.save(mask_path)
    return mask_path


def extract_line_polygons(example: dict) -> list:
    """
    Extract line-level polygons from a CATMuS segmentation example.
    """
    objects = example.get("objects") or {}
    polygons = objects.get("polygons", [])
    object_types = objects.get("type", [])

    # Extract only polygons corresponding to "line" objects, and filter out empty polygons
    return [
        polygon
        for polygon, object_type in zip(polygons, object_types)
        if object_type == "line" and polygon
    ]


def generate_line_mask_from_polygons(image_size: tuple[int, int], polygons: list) -> np.ndarray:
    """
    Generate a binary line mask from CATMuS polygons.

    Args:
        image_size: PIL image size as (width, height).
        polygons: CATMuS line polygons in image coordinates.
    """
    mask = PILImage.new("L", image_size, 0) # Create a blank mask image (grayscale, 0=black) with the same size as the original image
    draw = ImageDraw.Draw(mask)

    for polygon in polygons:
        if polygon:
            draw.polygon(polygon, outline=1, fill=1)

    return np.array(mask) # mask (height, width) with values in {0, 1}


def pil_to_rgb(image: PILImage.Image | np.ndarray) -> PILImage.Image:
    """
    Convert a PIL image or numpy array to a RGB PIL image.
    """
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")

    return PILImage.fromarray(np.asarray(image)).convert("RGB")


def preprocess_image(
    image: PILImage.Image | np.ndarray,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """
    Resize an image to (height, width, 3), normalize it to [0, 1], and return float32.
    Args:
        image: PIL image or numpy array.
        target_size: Desired output size as (height, width).
    Returns:
        Preprocessed image as a numpy array of shape (height, width, 3) with
        float32 values in [0, 1].
    
    """
    # This ensures that the image has three color channels (Red, Green, Blue) => 3 in shape, regardless of its original format.
    image_rgb = pil_to_rgb(image) # Convert the input image to RGB format. 
    image_resized = image_rgb.resize(target_size_to_pil_size(target_size), PILImage.Resampling.BILINEAR)
    image_preprocessed = np.asarray(image_resized, dtype=np.float32) / 255.0 # Convert the resized image to a numpy array and normalize pixel values to [0, 1].
    
    return image_preprocessed # Image in shape (height, width, 3) with float32 values in [0, 1]


def preprocess_binary_mask(
    mask: np.ndarray,
    target_size: tuple[int, int] | None = DEFAULT_IMAGE_SIZE,
    add_channel: bool = True,
) -> np.ndarray:
    """
    Resize a binary mask with nearest-neighbor interpolation and return float32 values in {0, 1}.
    """
    mask_array = (np.squeeze(mask) > 0).astype(np.uint8) * 255
    mask_image = PILImage.fromarray(mask_array, mode="L")

    if target_size is not None:
        mask_image = mask_image.resize(target_size_to_pil_size(target_size), PILImage.Resampling.NEAREST)

    mask_preprocessed = (np.asarray(mask_image) > 0).astype(np.float32)

    if add_channel:
        mask_preprocessed = mask_preprocessed[..., np.newaxis] # Add a channel dimension to the mask, resulting in shape (height, width, 1) 

    return mask_preprocessed # mask in shape (height, width, 1) with float32 values in {0, 1}


def preprocess_image_and_mask(
    example: dict,
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert one CATMuS example into a U-Net-ready pair:
    image=(height, width, 3), mask=(height, width, 1).
    """
    image = example["image"]
    image_array = preprocess_image(image, target_size=target_size)

    line_polygons = extract_line_polygons(example)
    mask = generate_line_mask_from_polygons(image.size, line_polygons)
    mask_array = preprocess_binary_mask(mask, target_size=target_size, add_channel=True)

    return image_array, mask_array


def load_segmentation_dataset(dataset_path: str | None = None):
    """
    Load the CATMuS segmentation dataset saved with HuggingFace save_to_disk.
    """
    dataset_path = dataset_path or os.path.join(data_dir, "segment_data")
    return load_data_from_dir(dataset_path)


def select_split_subset(split_dataset, max_examples: int | None = None):
    """
    Select the first max_examples items from a HuggingFace split when requested.
    """
    if max_examples is None or int(max_examples) <= 0:
        return split_dataset

    max_examples = min(int(max_examples), len(split_dataset))
    return split_dataset.select(range(max_examples))


def iter_preprocessed_images_and_masks(
    split: str = "train",
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    max_examples: int | None = None,
    dataset_path: str | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Stream preprocessed image/mask pairs without loading the whole split into RAM.
    """
    dataset = load_segmentation_dataset(dataset_path)
    split_dataset = select_split_subset(dataset[split], max_examples=max_examples)

    for example in split_dataset:
        yield preprocess_image_and_mask(example, target_size=target_size)


def preprocess_images_and_masks(
    split: str = "train",
    target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    max_examples: int | None = None,
    dataset_path: str | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Eagerly preprocess a split into lists. 
    Useful for tiny debug subsets and visualization.
    """
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for image, mask in iter_preprocessed_images_and_masks(
        split=split,
        target_size=target_size,
        max_examples=max_examples,
        dataset_path=dataset_path,
    ):
        images.append(image)
        masks.append(mask)

    return images, masks



# ------ Teste simple pour générer quelques masques de lignes 
# et sauvegarder dans le disque (présentation résultat d'une image de masque de lignes)
def batch_generate_lines_masks(
    dataset,
    split: str = "train",
    is_save: bool = False,
    save_dir: str | None = None,
    unique_names: bool = True,
    max_examples: int | None = None,
    target_size: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """
    Generate line masks for a CATMuS segmentation split.

    If target_size is provided, returned masks are resized to (height, width, 1).
    Otherwise returned masks keep original page size and remain 2D.
    """
    masks = []
    raw_dataset = dataset.cast_column("image", HFImage(decode=False))
    split_dataset = select_split_subset(dataset[split], max_examples=max_examples)
    raw_split_dataset = select_split_subset(raw_dataset[split], max_examples=max_examples)

    for i, example in enumerate(split_dataset):
        raw_example = raw_split_dataset[i]
        image_file_name = get_image_file_name(
            raw_example["image"],
            fallback_name=f"{split}_{i:06d}.png",
        )
        mask_file_name = get_mask_file_name(image_file_name)
        if unique_names:
            mask_file_name = f"{split}_{i:06d}_{mask_file_name}"

        line_polygons = extract_line_polygons(example)
        mask = generate_line_mask_from_polygons(example["image"].size, line_polygons)

        if is_save and save_dir:
            save_masks_to_disk(mask, save_dir, mask_file_name)

        if target_size is None:
            masks.append(preprocess_binary_mask(mask, target_size=None, add_channel=False))
        else:
            masks.append(preprocess_binary_mask(mask, target_size=target_size, add_channel=True))

    return masks



# Génération des masques de lignes pour 5 images du split "train", 
# juste pour montrer le résultat et vérifier que le pipeline fonctionne correctement.
# Sinon, le pipeline gère la génération des masques 
# à la volée pendant l'entrainement, sans stocker les masques sur le disque.
# donc les fonctions `batch_generate_lines_masks`, `save_masks_to_disk` 
# et `main_generate_line_masks` sont surtout pour le debug et la visualisation.
def main_generate_line_masks() -> None:
    """
    Generate and visualize a few line masks for quick debugging.
    """
    dataset = load_segmentation_dataset()
    output_mask_dir = os.path.join(outputs_dir, "segmentation", "masks")

    for split in ["train"]:
        output_dir_split = os.path.join(output_mask_dir, split)
        os.makedirs(output_dir_split, exist_ok=True)
        print(f"Generating masks for {split} split...")

        masks = batch_generate_lines_masks(
            dataset,
            split=split,
            is_save=True,
            save_dir=output_dir_split,
            unique_names=True,
            max_examples=5,
            target_size=DEFAULT_IMAGE_SIZE,
        )
        print(f"Masks for {split} split saved to {output_dir_split}.")

        fig, axes = plt.subplots(1, len(masks), figsize=(15, 3))
        if len(masks) == 1:
            axes = [axes]

        for axis, mask in zip(axes, masks):
            axis.imshow(np.squeeze(mask), cmap="gray")
            axis.axis("off")

        plt.suptitle(f"Generated Masks for {split} Split")
        plt.show()


if __name__ == "__main__":
    main_generate_line_masks()

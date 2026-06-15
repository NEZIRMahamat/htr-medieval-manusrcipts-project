"""
Mask generation utilities for CATMuS segmentation and U-Net training.
"""

import os

from scipy.ndimage import label, find_objects
import numpy as np
from datasets import Image as HFImage
from PIL import Image as PILImage, ImageDraw
from matplotlib import pyplot as plt

try:
    from .dataset import data_dir, outputs_dir, load_data_from_dir
except ImportError:
    from dataset import data_dir, outputs_dir, load_data_from_dir

## ------ Génération des masques pour la segmentation des lignes manuscrites dans le dataset CATMuS.

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
    mask_image = PILImage.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(mask_path)
    return mask_path


def generate_line_mask_from_polygons(image_size: tuple, polygons: list) -> np.ndarray:
    """
    Generate a binary mask from CATMuS polygons for a given image size.
    """
    mask = PILImage.new("L", image_size, 0) # New mask image avec fond noir (0), mode "L" pour niveaux de gris
    draw = ImageDraw.Draw(mask) # Create a drawing context for the mask image

    for polygon in polygons:
        # Draw the polygon on the mask image. The polygon is filled with white (1) to indicate the region of interest.
        if polygon:
            draw.polygon(polygon, outline=1, fill=1)

    return np.array(mask)


def batch_generate_lines_masks(
    dataset,
    split: str = "train",
    is_save: bool = False,
    save_dir: str | None = None,
    unique_names: bool = True,
    max_examples: int | None = None,
) -> list[np.ndarray]:
    """
    Generate masks for all images in a CATMuS segmentation split.
    """
    masks = []
    raw_dataset = dataset.cast_column("image", HFImage(decode=False))
    split_dataset = dataset[split]
    raw_split_dataset = raw_dataset[split]

    if max_examples is not None:
        max_examples = min(max_examples, len(split_dataset))
        split_dataset = split_dataset.select(range(max_examples))
        raw_split_dataset = raw_split_dataset.select(range(max_examples))

    for i, example in enumerate(split_dataset):
        raw_example = raw_split_dataset[i]
        image_file_name = get_image_file_name(
            raw_example["image"],
            fallback_name=f"{split}_{i:06d}.png",
        )
        mask_file_name = get_mask_file_name(image_file_name)
        if unique_names:
            mask_file_name = f"{split}_{i:06d}_{mask_file_name}"

        image_size = example["image"].size # (width, height)
        objects = example.get("objects", [])
        polygons = [ 
            polygon for polygon, object_type in
            zip(objects.get("polygons", []), objects.get("type", []))
            if object_type == "line"
        ]
        mask = generate_line_mask_from_polygons(image_size, polygons)
        
        
        masks.append(preprocess_binary_mask(mask))  # (Modèle U-Net expects float32 masks)

        if is_save and save_dir:
            save_masks_to_disk((mask > 0).astype(np.uint8), save_dir, mask_file_name)

    return masks


## ------ Mask Layout (Marge, Illustration, etc.) plus tard.


## ------ Pretraitement des images et masques pour l'entrainement du modèle U-Net.

# Détection des cas critiques (recours à la fin si modèle ne donne pas de bons résultats)
def detect_saturated_zones(
    image: np.ndarray,
    threshold_saturation    : int = 250,
    surface_min             : int = 100
) -> list[dict]:
    
    """
    Détecte les zones saturées dans une image (pixels très clairs) et retourne leurs coordonnées.
    Les zones saturées sont définies comme des régions où tous les canaux de couleur 
    dépassent un certain seuil (threshold_saturation).
    Retourne la liste des régions saturées avec leurs coordonnées (polygones).

    Args:
        image (np.ndarray): Image d'entrée (H, W, C) en uint8
        threshold_saturation (int): Seuil de saturation pour détecter les pixels saturés.
        surface_min (int): Surface minimale pour considérer une zone comme saturée.

    """
    mask_sature = np.all(image >= threshold_saturation, axis=2).astype(np.uint8) * 255

    # Label connected components in the binary mask to identify distinct saturated zones    
    labeled_array, num_features = label(mask_sature)
    
    regions_crtitiques = []
    for i in range(1, num_features + 1):
        zone = (labeled_array == i)
        if np.sum(zone) >= surface_min:
            coords = np.argwhere(zone)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            regions_crtitiques.append({
                "bbox": (x_min, y_min, x_max, y_max),
                "surface": np.sum(zone)
            })

    return regions_crtitiques


def preprocess_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Preprocess a binary mask for U-Net training.
    Convert to float32 and ensure values are 0 or 1.
    """
    return (mask > 0).astype(np.float32)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for U-Net training.
    Convert to float32 and normalize pixel values to [0, 1].
    """
    return image.astype(np.float32) / 255.0

def preprocess_images_and_masks(split: str = "train") -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Preprocess images and masks from a CATMuS segmentation dataset split for U-Net training.
    Returns lists of preprocessed images and masks.
    """
    dataset = load_data_from_dir(os.path.join(data_dir, "segment_data"))
    images = []
    masks = batch_generate_lines_masks(dataset, split=split, is_save=False)  # Generate masks without saving
    for example in dataset[split]:
        image = np.array(example["image"])
        images.append(preprocess_image(image))
    
    return images, masks



## ------ Main, test rapide et debugging développement.
def main_generate_line_masks() -> None:
    """
    Generate and save masks for all splits of the CATMuS segmentation dataset.
    Test génération et visualisation des 5 masks du dataset test pour vérifier 
    la qualité des masques générés. (train seulement, test rapide)
    """
    # Load the dataset from disk
    dataset_path = os.path.join(data_dir, "segment_data")
    dataset = load_data_from_dir(dataset_path)

    output_mask_dir = os.path.join(outputs_dir, "segmentation", "masks")

    os.makedirs(output_mask_dir, exist_ok=True)
    # Generate and save masks for each split
    for split in ["train"]: # , "validation", "test"
        output_dir_split = os.path.join(output_mask_dir, split)
        os.makedirs(output_dir_split, exist_ok=True)
        print(f"Generating masks for {split} split...")
        save_dir = output_dir_split
        masks = batch_generate_lines_masks(
            dataset,
            split=split,
            is_save=True,
            save_dir=save_dir,
            unique_names=True,
            max_examples=5,
        )
        print(f"Masks for {split} split saved to {save_dir}.")

        # Plot 
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for j, mask in enumerate(masks):
            axes[j].imshow(mask, cmap="gray")
            axes[j].axis("off")
        plt.suptitle(f"Generated Masks for {split} Split (First 5 Examples)")
        plt.show()


if __name__ == "__main__":
    main_generate_line_masks()

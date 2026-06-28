"""
Module for generating line crops from segmentation predictions.
This module provides functions to generate line crops from segmentation predictions JSON files.
Objectives:
- Load segmentation predictions and the corresponding dataset.
- Crop line images from the original page images based on predicted bounding boxes.
- Save the cropped line images and enrich the predictions JSON with crop metadata.
The enriched predictions JSON will include the following additional fields for each line:
- crop_path: Path to the saved crop image.
- crop_bbox: Bounding box of the crop in the original page image.
- crop_width: Width of the crop image.
- crop_height: Height of the crop image.

The module also includes a main function to run the crop generation process with hardcoded 
parameters for development purposes. 
The main function can be modified or replaced with a more flexible interface as needed.

"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

from PIL import Image as PILImage

try:
    from src.utils import load_segmentation_dataset
except ImportError:
    from utils import load_segmentation_dataset



def clamp_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
    margin: int = 0,
) -> list[int]:
    """
    Clamp a bbox to image bounds and optionally add a margin.
    """
    x_min, y_min, x_max, y_max = bbox

    return [
        max(0, int(x_min) - margin),
        max(0, int(y_min) - margin),
        min(image_width, int(x_max) + margin),
        min(image_height, int(y_max) + margin),
    ]


def is_valid_bbox(bbox: list[int]) -> bool:
    """
    Check that a bbox has positive width and height.
    """
    x_min, y_min, x_max, y_max = bbox
    return x_max > x_min and y_max > y_min


def crop_line_image(
    page_image: PILImage.Image,
    bbox: list[int],
    margin: int = 8,
) -> tuple[PILImage.Image, list[int]]:
    """
    Crop one line image from a page using bbox coordinates.
    """
    crop_bbox = clamp_bbox(
        bbox=bbox,
        image_width=page_image.width,
        image_height=page_image.height,
        margin=margin,
    )

    if not is_valid_bbox(crop_bbox):
        raise ValueError(f"Invalid bbox after clamping: {crop_bbox}")

    return page_image.crop(tuple(crop_bbox)), crop_bbox


def load_predictions(predictions_path: str) -> dict[str, Any]:
    """
    Load segmentation predictions JSON.
    """
    with open(predictions_path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def save_predictions(predictions: dict[str, Any], output_path: str) -> None:
    """
    Save enriched predictions JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(predictions, output_file, ensure_ascii=False, indent=2)


def get_page_image(dataset, split: str, split_index: int) -> PILImage.Image:
    """
    Load the original page image from the HuggingFace DatasetDict.
    """
    image = dataset[split][split_index]["image"]
    return image.convert("RGB")


def generate_crops_from_predictions(
    predictions_path: str,
    dataset_path: str,
    output_crops_root: str,
    enriched_predictions_output: str,
    crop_margin: int = 8,
) -> dict[str, Any]:
    """
    Generate line crops from a segmentation predictions JSON.

    The returned/saved JSON keeps the original prediction structure and adds crop metadata
    to each line:
        crop_path, crop_bbox, crop_width, crop_height.
    """
    predictions = load_predictions(predictions_path)
    enriched_predictions = deepcopy(predictions)
    dataset = load_segmentation_dataset(dataset_path)
    total_crops = 0

    for page in enriched_predictions.get("pages", []):
        split = page["split"]
        split_index = page["split_index"]
        page_id = page["page_id"]
        page_image = get_page_image(dataset, split=split, split_index=split_index)
        page_crop_dir = os.path.join(output_crops_root, split, page_id)
        os.makedirs(page_crop_dir, exist_ok=True)

        for line in page.get("lines", []):
            line_id = line["line_id"]
            crop_image, crop_bbox = crop_line_image(
                page_image=page_image,
                bbox=line["bbox"],
                margin=crop_margin,
            )

            crop_filename = f"{line_id}.png"
            crop_path = os.path.join(page_crop_dir, crop_filename)
            crop_image.save(crop_path)

            line["crop_path"] = crop_path.replace("\\", "/")
            line["crop_bbox"] = crop_bbox
            line["crop_width"] = crop_image.width
            line["crop_height"] = crop_image.height
            total_crops += 1

    enriched_predictions.setdefault("metadata", {})
    enriched_predictions["metadata"]["crop_generation"] = {
        "crop_margin_pixels": crop_margin,
        "output_crops_root": output_crops_root.replace("\\", "/"),
        "total_crops": total_crops,
    }

    save_predictions(enriched_predictions, enriched_predictions_output)
    return enriched_predictions


def main_generate_crops() -> None:
    """
    Generate line crops with hardcoded development parameters.
    """
    dataset_path = os.path.join("data", "segment_data")
    split = "test"
    predictions_path = os.path.join(
        "outputs",
        "segmentation",
        "predictions",
        f"{split}_segmentation_predictions.json",
    )
    output_crops_root = os.path.join("outputs", "segmentation", "crops_line")
    enriched_predictions_output = os.path.join(
        "outputs",
        "segmentation",
        "predictions",
        f"{split}_segmentation_predictions_with_crops.json",
    )
    crop_margin = 8

    enriched_predictions = generate_crops_from_predictions(
        predictions_path=predictions_path,
        dataset_path=dataset_path,
        output_crops_root=output_crops_root,
        enriched_predictions_output=enriched_predictions_output,
        crop_margin=crop_margin,
    )

    total_crops = enriched_predictions["metadata"]["crop_generation"]["total_crops"]
    print(f"Crops saved to: {os.path.join(output_crops_root, split)}")
    print(f"Enriched predictions saved to: {enriched_predictions_output}")
    print(f"Total crops generated: {total_crops}")


if __name__ == "__main__":
    main_generate_crops()

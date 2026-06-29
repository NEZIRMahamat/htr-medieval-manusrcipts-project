"""
Module for evaluating a trained U-Net model on the CATMuS line segmentation dataset.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from datasets import Image as HFImage
from scipy.ndimage import find_objects, label

try:
    from src.segmentation.model import combined_loss, dice_coefficient, mean_iou
    from src.segmentation.utils import (
        DEFAULT_IMAGE_SIZE,
        get_image_file_name,
        load_segmentation_dataset,
        preprocess_image_and_mask,
        select_split_subset,
    )
except ImportError:
    from model import combined_loss, dice_coefficient, mean_iou
    from utils import (
        DEFAULT_IMAGE_SIZE,
        get_image_file_name,
        load_segmentation_dataset,
        preprocess_image_and_mask,
        select_split_subset,
    )



def load_trained_model(model_path: str) -> tf.keras.Model:
    """
    Load a trained Keras model for inference/evaluation.
    """
    custom_objects = {
        "combined_loss": combined_loss,
        "dice_coefficient": dice_coefficient,
        "mean_iou": mean_iou,
    }
    return tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
    )


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute pixel-level binary segmentation metrics from thresholded masks.
    Returns a dictionary with dice, iou, precision, recall, and f1 scores. 
    (stats are computed on the binary masks, not the probability masks)
    """
    y_true_bool = np.asarray(y_true).astype(bool)
    y_pred_bool = np.asarray(y_pred).astype(bool)

    true_positive = np.logical_and(y_true_bool, y_pred_bool).sum()
    false_positive = np.logical_and(~y_true_bool, y_pred_bool).sum()
    false_negative = np.logical_and(y_true_bool, ~y_pred_bool).sum()
    union = np.logical_or(y_true_bool, y_pred_bool).sum()
    true_sum = y_true_bool.sum()
    pred_sum = y_pred_bool.sum()

    if true_sum == 0 and pred_sum == 0:
        return {
            "dice": 1.0,
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

    dice = (2 * true_positive) / (true_sum + pred_sum) if (true_sum + pred_sum) > 0 else 0.0
    iou = true_positive / union if union > 0 else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def rescale_bbox(
    bbox: tuple[int, int, int, int],
    model_size: tuple[int, int],
    original_size: tuple[int, int],
) -> list[int]:
    """
    Rescale a bbox from model coordinates to original page coordinates.

    Args:
        bbox: (x_min, y_min, x_max, y_max) in model-size pixels.
        model_size: (height, width) used by the model.
        original_size: PIL size (width, height) of the original page.
    Returns:
        List of [x_min, y_min, x_max, y_max] in original page coordinates.
    """
    x_min, y_min, x_max, y_max = bbox
    model_height, model_width = model_size
    original_width, original_height = original_size

    scale_x = original_width / model_width
    scale_y = original_height / model_height

    return [
        int(round(x_min * scale_x)),
        int(round(y_min * scale_y)),
        int(round(x_max * scale_x)),
        int(round(y_max * scale_y)),
    ]


def bbox_to_polygon(bbox: list[int]) -> list[list[int]]:
    """
    Convert [x_min, y_min, x_max, y_max] to a rectangular polygon.
    """
    x_min, y_min, x_max, y_max = bbox
    return [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]


def should_split_component_bbox(bbox: list[int], min_height: int = 300, min_ratio: float = 1.4) -> bool:
    """
    Decide if a predicted bbox is probably a text column/block rather than one line.
    """
    x_min, y_min, x_max, y_max = bbox
    width = max(1, x_max - x_min)
    height = max(1, y_max - y_min)
    return height >= min_height and (height / width) >= min_ratio


def split_bbox_into_text_lines(
    page_image,
    bbox: list[int],
    row_threshold_ratio: float = 0.25,
    dark_percentile: float = 80.0,
    min_line_height: int = 12,
    max_line_gap: int = 10,
    x_margin: int = 4,
) -> list[list[int]]:
    """
    Split a tall bbox into line bboxes using horizontal projection on the original image.
    """
    x_min, y_min, x_max, y_max = bbox
    crop_gray = page_image.crop((x_min, y_min, x_max, y_max)).convert("L")
    crop_array = np.asarray(crop_gray)

    if crop_array.size == 0:
        return []

    darkness = 255 - crop_array
    dark_threshold = max(35.0, float(np.percentile(darkness, dark_percentile)))
    text_mask = darkness >= dark_threshold
    row_projection = text_mask.sum(axis=1)

    if row_projection.max() == 0:
        return []

    row_threshold = max(8.0, float(row_projection.max()) * row_threshold_ratio)
    active_rows = row_projection >= row_threshold
    row_groups = collect_active_ranges(active_rows, min_length=min_line_height)
    row_groups = merge_close_ranges(row_groups, max_gap=max_line_gap)

    line_bboxes: list[list[int]] = []
    for row_start, row_end in row_groups:
        band_mask = text_mask[row_start:row_end, :]
        col_projection = band_mask.sum(axis=0)
        active_cols = col_projection > 0

        if active_cols.any():
            active_indices = np.where(active_cols)[0]
            local_x_min = max(0, int(active_indices[0]) - x_margin)
            local_x_max = min(crop_array.shape[1], int(active_indices[-1]) + 1 + x_margin)
        else:
            local_x_min = 0
            local_x_max = crop_array.shape[1]

        line_bbox = [
            x_min + local_x_min,
            y_min + int(row_start),
            x_min + local_x_max,
            y_min + int(row_end),
        ]

        if line_bbox[2] > line_bbox[0] and line_bbox[3] > line_bbox[1]:
            line_bboxes.append(line_bbox)

    return line_bboxes


def collect_active_ranges(active_values: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """
    Collect contiguous active ranges as (start, end).
    """
    ranges: list[tuple[int, int]] = []
    start = None

    for index, is_active in enumerate(active_values):
        if is_active and start is None:
            start = index

        is_last = index == len(active_values) - 1
        if start is not None and (not is_active or is_last):
            end = index if not is_active else index + 1
            if end - start >= min_length:
                ranges.append((start, end))
            start = None

    return ranges


def merge_close_ranges(ranges: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """
    Merge ranges separated by a small vertical gap.
    """
    if not ranges:
        return []

    merged = [ranges[0]]
    for start, end in ranges[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= max_gap:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))

    return merged


def assign_reading_order(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Assign reading order with a simple multi-column strategy.
    """
    columns: list[dict[str, Any]] = []

    for line in sorted(lines, key=lambda item: item["bbox"][0]):
        assigned_column = None
        line_bbox = line["bbox"]

        for column in columns:
            if horizontal_overlap_ratio(line_bbox, column["bbox"]) >= 0.3:
                assigned_column = column
                break

        if assigned_column is None:
            assigned_column = {"bbox": line_bbox.copy(), "lines": []}
            columns.append(assigned_column)
        else:
            assigned_column["bbox"] = merge_bboxes(assigned_column["bbox"], line_bbox)

        assigned_column["lines"].append(line)

    ordered_lines: list[dict[str, Any]] = []
    for column in sorted(columns, key=lambda item: item["bbox"][0]):
        ordered_lines.extend(sorted(column["lines"], key=lambda item: (item["bbox"][1], item["bbox"][0])))

    for reading_order, line in enumerate(ordered_lines, start=1):
        line["line_id"] = f"line_{reading_order:04d}"
        line["reading_order"] = reading_order

    return ordered_lines


def horizontal_overlap_ratio(bbox_a: list[int], bbox_b: list[int]) -> float:
    """
    Compute horizontal overlap over the smaller bbox width.
    """
    overlap = max(0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]))
    min_width = max(1, min(bbox_a[2] - bbox_a[0], bbox_b[2] - bbox_b[0]))
    return overlap / min_width


def merge_bboxes(bbox_a: list[int], bbox_b: list[int]) -> list[int]:
    """
    Merge two bboxes.
    """
    return [
        min(bbox_a[0], bbox_b[0]),
        min(bbox_a[1], bbox_b[1]),
        max(bbox_a[2], bbox_b[2]),
        max(bbox_a[3], bbox_b[3]),
    ]


def predicted_mask_to_lines(
    prediction_prob: np.ndarray,
    threshold: float,
    original_size: tuple[int, int],
    min_area: int,
    border_margin: int = 0,
    page_image=None,
) -> list[dict[str, Any]]:
    """
    Convert a predicted probability mask into line components.

    This version uses connected components. Tall components can be split into
    line bands using the original image, which helps with multi-column pages.
    """
    prediction_prob = np.squeeze(prediction_prob)
    binary_mask = prediction_prob >= threshold
    labeled_mask, num_components = label(binary_mask)
    component_slices = find_objects(labeled_mask)
    model_size = prediction_prob.shape
    lines: list[dict[str, Any]] = []

    for component_id in range(1, num_components + 1):
        component_slice = component_slices[component_id - 1]
        if component_slice is None:
            continue

        y_slice, x_slice = component_slice
        component_mask = labeled_mask[component_slice] == component_id
        area = int(component_mask.sum())

        if area < min_area:
            continue

        x_min, x_max = int(x_slice.start), int(x_slice.stop)
        y_min, y_max = int(y_slice.start), int(y_slice.stop)

        if touches_model_border(
            bbox=(x_min, y_min, x_max, y_max),
            model_size=model_size,
            border_margin=border_margin,
        ):
            continue

        bbox = rescale_bbox(
            bbox=(x_min, y_min, x_max, y_max),
            model_size=model_size,
            original_size=original_size,
        )
        score = float(prediction_prob[component_slice][component_mask].mean())

        if page_image is not None and should_split_component_bbox(bbox):
            line_bboxes = split_bbox_into_text_lines(page_image=page_image, bbox=bbox)
        else:
            line_bboxes = [bbox]

        for line_bbox in line_bboxes:
            lines.append(
                {
                    "bbox": line_bbox,
                    "polygon": bbox_to_polygon(line_bbox),
                    "score": score,
                    "area_model_pixels": area,
                    "source_component_bbox": bbox,
                }
            )

    return assign_reading_order(lines)


def touches_model_border(
    bbox: tuple[int, int, int, int],
    model_size: tuple[int, int],
    border_margin: int,
) -> bool:
    """
    Return True when a model-space bbox touches the model border margin.
    """
    if border_margin <= 0:
        return False

    x_min, y_min, x_max, y_max = bbox
    model_height, model_width = model_size

    return (
        x_min <= border_margin
        or y_min <= border_margin
        or x_max >= model_width - border_margin
        or y_max >= model_height - border_margin
    )


def page_identifier(split: str, index: int, raw_example: dict) -> tuple[str, str]:
    """
    Build a stable page id and image name from a raw HuggingFace example.
    """
    image_name = get_image_file_name(raw_example["image"], fallback_name=f"{split}_{index:06d}.png")
    stem = Path(image_name).stem
    return f"{split}_{index:06d}_{stem}", image_name


def evaluate_model(
    model_path: str,
    dataset_path: str,
    split: str = "validation",
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    max_examples: int | None = 20,
    threshold: float = 0.5,
    min_area: int = 20,
    border_margin: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Evaluate a trained segmentation model and build intermediate line predictions.
    """
    model = load_trained_model(model_path)
    dataset = load_segmentation_dataset(dataset_path)
    raw_dataset = dataset.cast_column("image", HFImage(decode=False))

    split_dataset = select_split_subset(dataset[split], max_examples=max_examples)
    raw_split_dataset = select_split_subset(raw_dataset[split], max_examples=max_examples)

    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
        "dataset_path": dataset_path,
        "split": split,
        "max_examples": max_examples,
        "image_size": {"height": image_size[0], "width": image_size[1]},
        "threshold": threshold,
        "min_area_model_pixels": min_area,
        "border_margin_model_pixels": border_margin,
        "postprocessing": "connected_components_with_column_line_split_v2",
    }

    page_metrics: list[dict[str, Any]] = []
    prediction_pages: list[dict[str, Any]] = []

    for index, example in enumerate(split_dataset):
        raw_example = raw_split_dataset[index]
        page_id, image_name = page_identifier(split, index, raw_example)

        image_array, gt_mask = preprocess_image_and_mask(example, target_size=image_size) # Preprocess the image and ground truth mask to the model's input size
        prediction_prob = model.predict(image_array[np.newaxis, ...], verbose=0)[0] # Predict the probability mask for the input image using the trained model
        prediction_binary = np.squeeze(prediction_prob) >= threshold # Convert the predicted probability mask to a binary mask using the specified threshold (0.5 by default)
        gt_binary = np.squeeze(gt_mask) >= 0.5 # Convert the ground truth mask to a binary mask (assuming the ground truth is in [0, 1] range)
        prediction_prob_squeezed = np.squeeze(prediction_prob) # Squeeze the predicted probability mask to remove any singleton dimensions for easier processing
        prediction_stats = {
            "min": float(prediction_prob_squeezed.min()),
            "max": float(prediction_prob_squeezed.max()),
            "mean": float(prediction_prob_squeezed.mean()),
        } # Stats about the predicted probability mask (min, max, mean) for analysis and debugging

        metrics = compute_binary_metrics(gt_binary, prediction_binary) # Compute binary metrics (e.g., precision, recall, F1-score) for the predicted mask
        lines = predicted_mask_to_lines(
            prediction_prob=prediction_prob,
            threshold=threshold,
            original_size=example["image"].size,
            min_area=min_area,
            border_margin=border_margin,
            page_image=example["image"],
        )

        page_metric = {
            "page_id": page_id,
            "image_name": image_name,
            "split_index": index,
            "predicted_line_count": len(lines),
            "predicted_line_pixel_count": int(prediction_binary.sum()),
            "gt_line_pixel_count": int(gt_binary.sum()),
            "prediction_prob_min": prediction_stats["min"],
            "prediction_prob_max": prediction_stats["max"],
            "prediction_prob_mean": prediction_stats["mean"],
            **metrics,
        }
        page_metrics.append(page_metric)

        original_width, original_height = example["image"].size
        prediction_pages.append(
            {
                "page_id": page_id,
                "image_name": image_name,
                "split": split,
                "split_index": index,
                "width": original_width,
                "height": original_height,
                "model_input_size": {"height": image_size[0], "width": image_size[1]},
                "prediction_stats": prediction_stats,
                "metrics": metrics,
                "lines": lines,
            }
        )

    mean_metrics = aggregate_metrics(page_metrics)
    metrics_payload = {
        "metadata": metadata,
        "mean_metrics": mean_metrics,
        "pages": page_metrics,
    }
    predictions_payload = {
        "metadata": metadata,
        "mean_metrics": mean_metrics,
        "pages": prediction_pages,
    }

    return metrics_payload, predictions_payload


def aggregate_metrics(page_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """
    Average page-level metrics.
    """
    if not page_metrics:
        return {}

    metric_names = ["dice", "iou", "precision", "recall", "f1"]
    return {
        metric_name: float(np.mean([page[metric_name] for page in page_metrics]))
        for metric_name in metric_names
    }


def save_json(payload: dict[str, Any], output_path: str) -> None:
    """
    Save a JSON payload with UTF-8 encoding.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def main_evaluate() -> None:
    """
    Run evaluation with hardcoded development parameters.
    """
    checkpoints_dir = os.path.join("outputs", "segmentation", "checkpoints")
    file_keras = os.listdir(checkpoints_dir)
    model_path = os.path.join(checkpoints_dir, file_keras[-1])  # Use the last saved model (.keras) checkpoint
    print(f"Evaluating model: {model_path}\n")
    dataset_path = os.path.join("data", "segment_data")
    split = "test"
    image_size = (256, 256)
    max_examples = int(0.1 * 600)  # 10% of the total dataset (600 examples) for evaluation
    threshold = 0.5
    min_area = 20 # Minimum area in model pixels to consider a connected component as a valid line.
    border_margin = 0 # Minimum distance in model pixels from the image border to consider a connected component as valid.

    metrics_output = os.path.join("outputs", "segmentation", "metrics", f"{split}_segmentation_metrics.json")
    predictions_output = os.path.join("outputs", "segmentation", "predictions", f"{split}_segmentation_predictions.json")

    metrics_payload, predictions_payload = evaluate_model(
        model_path=model_path,
        dataset_path=dataset_path,
        split=split,
        image_size=image_size,
        max_examples=max_examples,
        threshold=threshold,
        min_area=min_area,
        border_margin=border_margin,
    )

    save_json(metrics_payload, metrics_output)
    save_json(predictions_payload, predictions_output)

    print(f"Metrics saved to: {metrics_output}")
    print(f"Predictions saved to: {predictions_output}")
    print(f"Mean metrics: {metrics_payload['mean_metrics']}")


if __name__ == "__main__":
    main_evaluate()

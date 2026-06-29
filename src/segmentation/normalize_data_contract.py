"""
Module for normalizing segmentation predictions into a data contract format.
This module provides functions to normalize segmentation predictions into a standardized data contract format.
Objectives:
- Load segmentation predictions JSON files.
- Normalize the predictions and metadata into a structured data contract.
- Validate the normalized data contract against a predefined JSON schema.
The normalized data contract will include the following fields:
- schema_name: Name of the data contract schema.
- contract_version: Version of the data contract schema.
- dataset: Information about the dataset used for segmentation.
- generated_at: Timestamp of when the data contract was generated.
- source: Information about the source of the predictions.
- pipeline: Information about the segmentation pipeline used.
- summary: Summary statistics of the segmentation predictions.
- pages: List of pages with their corresponding lines and metadata.

The module also includes a main function to run the normalization process with hardcoded
parameters for development purposes. The main function can be modified or replaced with a more flexible interface as
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

try:
    from src.segmentation.data_contract_json import (
        DATA_CONTRACT_NAME,
        DATA_CONTRACT_VERSION,
        SEGMENTATION_DATA_CONTRACT_SCHEMA,
    )
except ImportError:
    from data_contract_json import (
        DATA_CONTRACT_NAME,
        DATA_CONTRACT_VERSION,
        SEGMENTATION_DATA_CONTRACT_SCHEMA,
    )


DATASET_NAME = "CATMuS/medieval-segmentation"
DATASET_TASK = "page-level line segmentation"


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as input_file:
        return json.load(input_file)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def normalize_path(path: Any) -> str | None:
    if path is None:
        return None

    normalized = str(path).replace("\\", "/")
    if normalized.startswith("outputs/checkpoints/"):
        normalized = normalized.replace("outputs/checkpoints/", "outputs/segmentation/checkpoints/", 1)
    return normalized


def read_text_if_exists(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8-sig").strip()


def to_int_list(values: Any) -> list[int]:
    return [int(value) for value in values]


def normalize_line(line: dict[str, Any]) -> dict[str, Any]:
    crop_path = normalize_path(line.get("crop_path"))
    if not crop_path:
        raise ValueError(f"Missing crop_path for line {line.get('line_id')}")

    return {
        "line_id": str(line["line_id"]),
        "reading_order": int(line["reading_order"]),
        "geometry": {
            "bbox": to_int_list(line["bbox"]),
            "polygon": [[int(x), int(y)] for x, y in line["polygon"]],
        },
        "segmentation": {
            "score": float(line["score"]),
            "area_model_pixels": int(line["area_model_pixels"]),
            "source_component_bbox": to_int_list(line["source_component_bbox"]),
        },
        "crop": {
            "path": crop_path,
            "bbox": to_int_list(line["crop_bbox"]),
            "width": int(line["crop_width"]),
            "height": int(line["crop_height"]),
        },
    }


def normalize_page(page: dict[str, Any]) -> dict[str, Any]:
    lines = [normalize_line(line) for line in page.get("lines", [])]

    return {
        "page_id": str(page["page_id"]),
        "image_name": str(page["image_name"]),
        "split": str(page["split"]),
        "split_index": int(page["split_index"]),
        "width": int(page["width"]),
        "height": int(page["height"]),
        "metrics": {
            "dice": float(page["metrics"]["dice"]),
            "iou": float(page["metrics"]["iou"]),
            "precision": float(page["metrics"]["precision"]),
            "recall": float(page["metrics"]["recall"]),
            "f1": float(page["metrics"]["f1"]),
        },
        "line_count": len(lines),
        "lines": lines,
    }


def normalize_segmentation_predictions(
    predictions_json_path: str | Path,
    output_json_path: str | Path,
    validate_output: bool = True,
) -> dict[str, Any]:
    predictions_json_path = Path(predictions_json_path)
    predictions = load_json(predictions_json_path)
    metadata = predictions["metadata"]
    pages = [normalize_page(page) for page in predictions.get("pages", [])]
    split = str(metadata["split"])

    crop_generation = metadata.get("crop_generation") or {}
    test_hash_path = Path("outputs/splits/test_set_segmentation_sha256.txt")
    test_manifest_path = Path("outputs/splits/test_set_segmentation_manifest.json")

    contract = {
        "schema_name": DATA_CONTRACT_NAME,
        "contract_version": DATA_CONTRACT_VERSION,
        "dataset": {
            "name": DATASET_NAME,
            "task": DATASET_TASK,
            "local_path": normalize_path(metadata.get("dataset_path")),
            "split": split,
            "max_examples": metadata.get("max_examples"),
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "predictions_json": normalize_path(predictions_json_path),
            "test_set_sha256": read_text_if_exists(test_hash_path) if split == "test" else None,
            "test_set_manifest": normalize_path(test_manifest_path) if split == "test" and test_manifest_path.exists() else None,
        },
        "pipeline": {
            "stage": "segmentation",
            "model": {
                "architecture": "U-Net binary line segmentation",
                "checkpoint_path": normalize_path(metadata.get("model_path")),
                "input_size": {
                    "height": int(metadata["image_size"]["height"]),
                    "width": int(metadata["image_size"]["width"]),
                },
            },
            "postprocessing": {
                "name": str(metadata.get("postprocessing")),
                "threshold": float(metadata["threshold"]),
                "min_area_model_pixels": int(metadata["min_area_model_pixels"]),
                "border_margin_model_pixels": int(metadata["border_margin_model_pixels"]),
            },
            "crop_generation": {
                "crop_margin_pixels": int(crop_generation.get("crop_margin_pixels", 0)),
                "output_crops_root": normalize_path(crop_generation.get("output_crops_root")),
                "total_crops": int(crop_generation.get("total_crops", 0)),
            },
        },
        "summary": {
            "page_count": len(pages),
            "line_count": sum(page["line_count"] for page in pages),
            "mean_metrics": {
                "dice": float(predictions["mean_metrics"]["dice"]),
                "iou": float(predictions["mean_metrics"]["iou"]),
                "precision": float(predictions["mean_metrics"]["precision"]),
                "recall": float(predictions["mean_metrics"]["recall"]),
                "f1": float(predictions["mean_metrics"]["f1"]),
            },
        },
        "pages": pages,
    }

    if validate_output:
        valid, errors = validate_contract_document(contract)
        if not valid:
            raise ValueError("Invalid segmentation contract:\n" + "\n".join(errors[:10]))

        missing_crops = find_missing_crop_paths(contract)
        if missing_crops:
            raise FileNotFoundError(
                "Some crop files referenced by the contract do not exist:\n"
                + "\n".join(missing_crops[:10])
            )

    save_json(contract, output_json_path)
    return contract


def validate_contract_document(document: dict[str, Any]) -> tuple[bool, list[str]]:
    validator = Draft202012Validator(SEGMENTATION_DATA_CONTRACT_SCHEMA)
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    formatted_errors = [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in errors
    ]
    return not formatted_errors, formatted_errors


def validate_data_json_file(path: str | Path) -> tuple[bool, list[str]]:
    return validate_contract_document(load_json(path))


def find_missing_crop_paths(document: dict[str, Any]) -> list[str]:
    missing_paths: list[str] = []

    for page in document.get("pages", []):
        for line in page.get("lines", []):
            crop_path = line["crop"]["path"]
            if not Path(crop_path).exists():
                missing_paths.append(crop_path)

    return missing_paths


def main_normalize_data_contract() -> None:
    output_dir = Path("outputs/segmentation/data_contract")
    input_paths = [
        Path("outputs/segmentation/predictions/validation_segmentation_predictions_with_crops.json"),
        Path("outputs/segmentation/predictions/test_segmentation_predictions_with_crops.json"),
    ]

    for input_path in input_paths:
        if not input_path.exists():
            print(f"Skipping missing input: {input_path}")
            continue

        split = input_path.name.split("_segmentation_predictions_with_crops.json")[0]
        output_path = output_dir / f"{split}_segmentation_contract.json"
        contract = normalize_segmentation_predictions(
            predictions_json_path=input_path,
            output_json_path=output_path,
            validate_output=True,
        )

        print(f"Contract saved to: {output_path}")
        print(
            f"Split={contract['dataset']['split']} "
            f"pages={contract['summary']['page_count']} "
            f"lines={contract['summary']['line_count']} "
            f"iou={contract['summary']['mean_metrics']['iou']:.4f}"
        )


if __name__ == "__main__":
    main_normalize_data_contract()

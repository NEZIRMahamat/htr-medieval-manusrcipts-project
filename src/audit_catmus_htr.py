from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


DEFAULT_DATASET_NAME = "CATMuS/medieval"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "train"
DEFAULT_REPORT_PATH = Path("DATA_SOURCES.md")
DATASETS_SERVER = "https://datasets-server.huggingface.co"


def get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def load_dataset_info(dataset_name: str) -> dict[str, Any]:
    """
    Load CATMuS metadata from Hugging Face without downloading the dataset.
    """
    return get_json(f"{DATASETS_SERVER}/info", {"dataset": dataset_name})


def load_first_rows(dataset_name: str, config: str, split: str) -> dict[str, Any]:
    """
    Load a small sample from Hugging Face Dataset Viewer.
    """
    return get_json(
        f"{DATASETS_SERVER}/first-rows",
        {"dataset": dataset_name, "config": config, "split": split},
    )


def compact_image_value(value: Any) -> Any:
    if isinstance(value, dict) and {"src", "height", "width"}.issubset(value):
        stable_src = value["src"].split("?", 1)[0]
        return {
            "type": "image",
            "height": value["height"],
            "width": value["width"],
            "src_without_temporary_signature": stable_src,
        }
    return value


def build_report(
    dataset_name: str,
    config: str,
    split: str,
    info_payload: dict[str, Any],
    rows_payload: dict[str, Any],
) -> str:
    dataset_info = info_payload["dataset_info"][config]
    features = dataset_info["features"]
    splits = dataset_info["splits"]
    sample_row = rows_payload["rows"][0]["row"]

    image_fields = [
        name
        for name, feature in features.items()
        if isinstance(feature, dict) and feature.get("_type") == "Image"
    ]
    transcription_fields = [
        name
        for name, feature in features.items()
        if name.lower() in {"text", "transcription", "label"}
        and isinstance(feature, dict)
        and feature.get("_type") == "Value"
    ]
    metadata_fields = [
        name
        for name in features
        if name not in set(image_fields + transcription_fields)
    ]

    split_counts = {name: split_info["num_examples"] for name, split_info in splits.items()}
    total_count = sum(split_counts.values())

    sample_preview = {
        key: compact_image_value(value)
        for key, value in sample_row.items()
    }

    lines = [
        "# CATMuS HTR Training Data Notes",
        "",
        f"Dataset: `{dataset_name}`",
        f"Config/subset: `{config}`",
        "Task: image-to-text handwritten text recognition.",
        "",
        "Source used for this audit:",
        f"- Hugging Face dataset metadata: `{DATASETS_SERVER}/info?dataset={dataset_name}`",
        f"- Hugging Face first rows: `{DATASETS_SERVER}/first-rows?dataset={dataset_name}&config={config}&split={split}`",
        "",
        "## Splits and Counts",
        "",
        "| Split | Samples |",
        "| --- | ---: |",
    ]

    for split_name, count in split_counts.items():
        lines.append(f"| `{split_name}` | {count:,} |")

    lines.extend(
        [
            f"| **Total** | **{total_count:,}** |",
            "",
            "## Fields",
            "",
            f"Image field: `{', '.join(image_fields)}`",
            f"Transcription field: `{', '.join(transcription_fields)}`",
            f"Metadata fields: `{', '.join(metadata_fields)}`",
            "",
            "For the HTR part, the model input is the line image in `im`, and the target text is `text`.",
            "",
            "## Feature Schema",
            "",
            "```json",
            json.dumps(features, indent=2, ensure_ascii=False),
            "```",
            "",
            "## One Training Sample",
            "",
            "```json",
            json.dumps(sample_preview, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Report Notes",
            "",
            "- CATMuS/medieval is already line-level HTR data: each row contains one text-line image and its transcription.",
            "- The line image column is `im`; the transcription/label column is `text`.",
            "- Useful metadata to preserve in JSON outputs: `language`, `century`, `region`, `script_type`, `shelfmark`, `verse`, `genre`, `project`, `line_type`, and `gen_split`.",
            "- The official Hugging Face split names are `train`, `validation`, and `test`.",
            "- The sample rows also contain `gen_split`, whose values can be `train`, `dev`, or `test`; keep it as metadata, but use the official split for training/evaluation separation.",
            "- Keep the `test` split sealed until final evaluation to avoid data leakage.",
            "",
        ]
    )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CATMuS HTR training data.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    info_payload = load_dataset_info(args.dataset_name)
    rows_payload = load_first_rows(args.dataset_name, args.config, args.split)

    report = build_report(
        dataset_name=args.dataset_name,
        config=args.config,
        split=args.split,
        info_payload=info_payload,
        rows_payload=rows_payload,
    )

    report_path = Path(args.report_path)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

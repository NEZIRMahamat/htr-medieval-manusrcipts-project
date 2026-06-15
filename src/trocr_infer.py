from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


DEFAULT_DATASET_NAME = "CATMuS/medieval"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "train"
DEFAULT_MODEL_NAME = "microsoft/trocr-base-handwritten"
DEFAULT_OUTPUT_DIR = Path("outputs") / "trocr_baseline"
DATASETS_SERVER = "https://datasets-server.huggingface.co"


def get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def load_catmus_rows(
    dataset_name: str,
    config: str,
    split: str,
    num_samples: int,
) -> list[dict[str, Any]]:
    """
    Load sample rows from Hugging Face Dataset Viewer without downloading CATMuS.
    """
    payload = get_json(
        f"{DATASETS_SERVER}/rows",
        {
            "dataset": dataset_name,
            "config": config,
            "split": split,
            "offset": 0,
            "length": num_samples,
        },
    )
    return [item["row"] for item in payload["rows"]]


def load_catmus_first_rows(
    dataset_name: str,
    config: str,
    split: str,
    num_samples: int,
) -> list[dict[str, Any]]:
    """
    Fallback endpoint for the first available sample rows.
    """
    payload = get_json(
        f"{DATASETS_SERVER}/first-rows",
        {
            "dataset": dataset_name,
            "config": config,
            "split": split,
        },
    )
    return [item["row"] for item in payload["rows"][:num_samples]]


def fetch_image(image_url: str) -> Image.Image:
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_trocr(model_name: str, device: torch.device):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        use_safetensors=False,
    )
    model.to(device)
    model.eval()
    return processor, model


def transcribe_image(
    image: Image.Image,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    device: torch.device,
    max_new_tokens: int = 128,
) -> str:
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
        )

    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return prediction.strip()


def save_image(image: Image.Image, output_dir: Path, index: int) -> str:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / f"catmus_line_{index:03d}.jpg"
    image.save(image_path)
    return str(image_path)


def build_markdown_report(results: list[dict[str, Any]], model_name: str) -> str:
    lines = [
        "# TrOCR Baseline Examples",
        "",
        f"Model: `{model_name}`",
        "",
    ]

    for result in results:
        lines.extend(
            [
                f"## Example {result['index']}",
                "",
                f"Image: `{result['image_path']}`",
                "",
                f"GT: {result['ground_truth']}",
                "",
                f"PR: {result['prediction']}",
                "",
                f"Metadata: `{json.dumps(result['metadata'], ensure_ascii=False)}`",
                "",
            ]
        )

    return "\n".join(lines)


def run_baseline(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows = load_catmus_rows(
            dataset_name=args.dataset_name,
            config=args.config,
            split=args.split,
            num_samples=args.num_samples,
        )
    except requests.HTTPError:
        rows = load_catmus_first_rows(
            dataset_name=args.dataset_name,
            config=args.config,
            split=args.split,
            num_samples=args.num_samples,
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    processor, model = load_trocr(args.model_name, device)

    results = []
    for index, row in enumerate(rows, start=1):
        image = fetch_image(row["im"]["src"])
        prediction = transcribe_image(
            image=image,
            processor=processor,
            model=model,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        image_path = save_image(image, output_dir, index)

        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"im", "text"}
        }
        result = {
            "index": index,
            "image_path": image_path,
            "ground_truth": row["text"],
            "prediction": prediction,
            "metadata": metadata,
        }
        results.append(result)
        print(f"Example {index}")
        print(f"GT: {result['ground_truth']}")
        print(f"PR: {result['prediction']}")
        print()

    json_path = output_dir / "trocr_baseline_examples.json"
    md_path = output_dir / "trocr_baseline_examples.md"

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(build_markdown_report(results, args.model_name), encoding="utf-8")

    print(f"Saved JSON examples to: {json_path}")
    print(f"Saved Markdown examples to: {md_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline TrOCR inference on CATMuS line images.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    run_baseline(parse_args())

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from jiwer import cer, wer

load_dotenv()
token = os.environ.get("HF_TOKEN")

DEFAULT_RESULTS_PATH = Path("outputs") / "trocr_baseline" / "trocr_baseline_examples.json"
DEFAULT_EVALUATION_PATH = Path("outputs") / "trocr_baseline" / "evaluation_results.json"


def load_results(results_path: Path) -> list[dict[str, Any]]:
    with results_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_results(results: list[dict[str, Any]]) -> tuple[float, float]:
    references = [result["ground_truth"] for result in results]
    predictions = [result["prediction"] for result in results]

    cer_score = cer(references, predictions)
    wer_score = wer(references, predictions)

    return cer_score, wer_score


def build_evaluation_report(
    results: list[dict[str, Any]],
    cer_score: float,
    wer_score: float,
) -> dict[str, Any]:
    examples = [
        {
            "index": result["index"],
            "image_path": result["image_path"],
            "ground_truth": result["ground_truth"],
            "prediction": result["prediction"],
            "metadata": result.get("metadata", {}),
        }
        for result in results
    ]

    return {
        "metrics": {
            "cer": cer_score,
            "wer": wer_score,
        },
        "num_examples": len(examples),
        "examples": examples,
    }


def save_evaluation_report(report: dict[str, Any], evaluation_path: Path) -> None:
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TrOCR baseline predictions with CER and WER.")
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--evaluation-path", default=str(DEFAULT_EVALUATION_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(Path(args.results_path))
    cer_score, wer_score = evaluate_results(results)
    report = build_evaluation_report(results, cer_score, wer_score)
    save_evaluation_report(report, Path(args.evaluation_path))

    print(f"Examples: {len(results)}")
    print(f"CER: {cer_score:.4f}")
    print(f"WER: {wer_score:.4f}")
    print(f"Saved evaluation results to: {args.evaluation_path}")


if __name__ == "__main__":
    main()

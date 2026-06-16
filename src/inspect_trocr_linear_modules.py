from __future__ import annotations

import argparse
import logging
import os

import torch
from dotenv import load_dotenv
from huggingface_hub.utils import logging as hf_logging
from transformers import VisionEncoderDecoderModel
from transformers.utils import logging as transformers_logging

load_dotenv()
token = os.environ.get("HF_TOKEN")

DEFAULT_MODEL_NAME = "microsoft/trocr-base-handwritten"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print TrOCR Linear module names for LoRA target selection.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    hf_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    model = VisionEncoderDecoderModel.from_pretrained(args.model_name, token=token)  # <-- token passed here

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(name)


if __name__ == "__main__":
    main()
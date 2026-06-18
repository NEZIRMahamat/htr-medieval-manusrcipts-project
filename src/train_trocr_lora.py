from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import torch
from datasets import Dataset, DatasetDict, load_dataset
from jiwer import cer, wer
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
from huggingface_hub import login
from dotenv import load_dotenv
import peft.utils.save_and_load as peft_save

load_dotenv()
DEFAULT_DATASET_NAME = "CATMuS/medieval"
DEFAULT_MODEL_NAME = "microsoft/trocr-base-handwritten"
DEFAULT_OUTPUT_DIR = Path("outputs") / "trocr_lora"
DEFAULT_LORA_RANKS = [8, 16]


@dataclass
class TrOCRDataCollator:
    processor: TrOCRProcessor
    max_target_length: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        from io import BytesIO
        images = [Image.open(BytesIO(feature["im"])).convert("RGB") for feature in features]
        texts = [feature["text"] for feature in features]

        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def select_samples(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def load_catmus_dataset(args: argparse.Namespace) -> DatasetDict:
    from io import BytesIO
    print("Loading dataset in streaming mode...")
    dataset = load_dataset(args.dataset_name, streaming=True)
    print("Dataset loaded.")

    train_data = dataset[args.train_split]
    eval_data = dataset[args.eval_split]

    if args.max_train_samples:
        train_data = train_data.take(args.max_train_samples)
    if args.max_eval_samples:
        eval_data = eval_data.take(args.max_eval_samples)

    def collect_samples(stream):
        print("  entering collect_samples...")
        samples = []
        print("  starting loop...")
        for i, sample in enumerate(stream):
            print(f"  got sample {i+1} from stream")
            img = sample["im"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            samples.append({
                "im": buf.getvalue(),  # store as bytes
                "text": sample["text"],
            })
            print(f"  loaded sample {i+1}")
        return samples

    print("Converting train split...")
    train_data = Dataset.from_list(collect_samples(train_data))
    print(f"Train ready: {len(train_data)} examples.")

    print("Converting eval split...")
    eval_data = Dataset.from_list(collect_samples(eval_data))
    print(f"Eval ready: {len(eval_data)} examples.")

    return DatasetDict({"train": train_data, "eval": eval_data})


def configure_model(model: VisionEncoderDecoderModel, processor: TrOCRProcessor) -> None:
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id


def find_lora_target_modules(
    model: VisionEncoderDecoderModel,
    candidate_modules: list[str],
) -> list[str]:
    found = {
        module_name.rsplit(".", 1)[-1]
        for module_name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and module_name.rsplit(".", 1)[-1] in set(candidate_modules)
    }
    if not found:
        raise ValueError(
            "No LoRA target modules were found. "
            f"Tried: {candidate_modules}. Inspect model.named_modules() and pass --target-modules."
        )
    return sorted(found)


def build_lora_model(
    model_name: str,
    processor: TrOCRProcessor,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
) -> VisionEncoderDecoderModel:
    model = VisionEncoderDecoderModel.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        use_safetensors=False,
    )
    configure_model(model, processor)

    resolved_target_modules = find_lora_target_modules(model, target_modules)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=resolved_target_modules,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_compute_metrics(processor: TrOCRProcessor):
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        pred_ids = eval_pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = eval_pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # clip pred_ids to valid vocabulary range
        vocab_size = processor.tokenizer.vocab_size
        pred_ids = pred_ids.clip(0, vocab_size - 1)

        predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)
        references = processor.batch_decode(label_ids, skip_special_tokens=True)

        return {
            "cer": cer(references, predictions),
            "wer": wer(references, predictions),
        }

    return compute_metrics


def train_one_rank(
    rank: int,
    dataset: DatasetDict,
    processor: TrOCRProcessor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_output_dir = Path(args.output_dir) / f"r_{rank}"
    model = build_lora_model(
        model_name=args.model_name,
        processor=processor,
        rank=rank,
        alpha=args.lora_alpha or rank * 2,
        dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy="no",        # <-- change this
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        seed=args.seed,
        use_cpu=args.cpu)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=TrOCRDataCollator(
            processor=processor,
            max_target_length=args.max_target_length,
        ),
        compute_metrics=build_compute_metrics(processor),
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    eval_metrics = trainer.evaluate()

    original_get_peft_model_state_dict = peft_save.get_peft_model_state_dict

    def patched_get_peft_model_state_dict(model, *args, **kwargs):
        if not hasattr(model.config, "vocab_size"):
            model.config.vocab_size = model.config.decoder.vocab_size
        return original_get_peft_model_state_dict(model, *args, **kwargs)

    peft_save.get_peft_model_state_dict = patched_get_peft_model_state_dict

    try:
        adapter_dir = run_output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(adapter_dir))
        processor.save_pretrained(str(run_output_dir / "processor"))

        print(f"Saved adapter to: {adapter_dir}")

    except Exception as e:
        print(f"Adapter save failed: {e}") 

    metrics = {
        "rank": rank,
        "output_dir": str(run_output_dir),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    (run_output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics


def save_comparison(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "lora_comparison.json"
    comparison_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved LoRA comparison to: {comparison_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on CATMuS with LoRA.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--lora-ranks", type=int, nargs="+", default=DEFAULT_LORA_RANKS)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["query", "value", "q_proj", "v_proj"],
        help="Linear module names to wrap with LoRA if present in the model.",
    )
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--generation-max-length", type=int, default=256)
    parser.add_argument("--generation-num-beams", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-strategy", default="steps")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-strategy", default="steps")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
            print("Logged into Hugging Face Hub using HF_TOKEN.")
        except Exception as exc:
            print(f"Warning: HF login failed: {exc}")

    print("Loading dataset...")  # load dataset first
    dataset = load_catmus_dataset(args)
    print("Dataset ready.")

    print("Loading processor...")  # then load processor
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    print("Processor loaded.")

    print("Starting training loop...")
    results = []
    for rank in args.lora_ranks:
        print(f"Starting LoRA fine-tuning with r={rank}")
        result = train_one_rank(rank, dataset, processor, args)
        results.append(result)

    save_comparison(results, output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

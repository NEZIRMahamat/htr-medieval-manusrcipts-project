from __future__ import annotations

import argparse
import importlib.util
import json
import time
from tqdm.auto import tqdm
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from .model import compile_unet_model
from .utils import (
    DEFAULT_IMAGE_SIZE,
    load_segmentation_dataset,
    iter_preprocessed_images_and_masks,
    select_split_subset,
    preprocess_images_and_masks, # pour train sur le dataset global
)


def configure_tensorflow() -> None:
    """
    Keep TensorFlow setup predictable across local CPU, Colab CPU, and optional GPU.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def make_tf_dataset(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    batch_size: int = 1,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from already preprocessed images and masks.

    This is useful only for tiny debug subsets.
    Prefer load_and_preprocess_data for training.
    """
    images_array = np.stack(images).astype(np.float32)
    masks_array = np.stack(masks).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((images_array, masks_array))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images_array), reshuffle_each_iteration=True)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_and_preprocess_data(
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    max_examples: int | None = None,
    dataset_path: str | None = None,
    running_mode: str = "debug",
) -> tf.data.Dataset:
    """
    Stream preprocessed CATMuS examples as TensorFlow batches.
    Args:
        split: Dataset split to load ("train", "validation", or "test").
        batch_size: Number of examples per batch.
        shuffle: Whether to shuffle the dataset (should be True for training, False for validation/test).
        image_size: Target size for images and masks as (height, width).
        max_examples: Maximum number of examples to load from the split (for debugging). None means no limit.
        dataset_path: Optional path to the preprocessed dataset on disk. If None, uses default loading logic.
        running_mode: "debug" or "full". In "debug" mode, it may use
            a smaller subset or different loading logic for faster iteration.
            In "full" mode, build dataset with all available examples (preprocess_images_and_masks).
    Output shapes:
        image: (height, width, 3), float32 in [0, 1]
        mask:  (height, width, 1), float32 in {0, 1}
    """
    if running_mode not in {"debug", "full"}:
        raise ValueError("running_mode must be 'debug' or 'full'.")

    if running_mode == "full":
        images, masks = preprocess_images_and_masks(
            split=split,
            target_size=image_size,
            max_examples=max_examples,
            dataset_path=dataset_path,
        )
        return make_tf_dataset(
            images=images,
            masks=masks,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    height, width = image_size
    output_signature = (
        tf.TensorSpec(shape=(height, width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(height, width, 1), dtype=tf.float32),
    )

    dataset = load_segmentation_dataset(dataset_path)
    split_dataset = select_split_subset(dataset[split], max_examples=max_examples)
    cardinality = len(split_dataset)

    if cardinality == 0:
        raise ValueError(f"No examples available for split '{split}'.")

    dataset_tf = tf.data.Dataset.from_generator(
        lambda: iter_preprocessed_images_and_masks(
            split=split,
            target_size=image_size,
            max_examples=max_examples,
            dataset_path=dataset_path,
        ),
        output_signature=output_signature,
    )
    dataset_tf = dataset_tf.apply(tf.data.experimental.assert_cardinality(cardinality))

    if shuffle:
        shuffle_buffer = min(cardinality, max_examples if max_examples is not None else 256)
        dataset_tf = dataset_tf.shuffle(
            buffer_size=max(1, shuffle_buffer),
            reshuffle_each_iteration=True,
        )

    return dataset_tf.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_val_dataset_preprocessed(    
    batch_size: int,
    image_size: tuple[int, int],
    max_train_examples: int | None,
    max_val_examples: int | None,
    dataset_path: str | None,
    running_mode: str = "debug",
) -> tf.data.Dataset:
    """
    Build separate preprocessed TensorFlow datasets for training and validation splits.
    This is useful for the "full" running mode where we preprocess the entire dataset at once.
     In "debug" mode, it may use different loading logic for faster iteration.
    """
    dataset_train = load_and_preprocess_data(
        split="train",
        batch_size=batch_size,
        shuffle=True,
        image_size=image_size,
        max_examples=max_train_examples,
        dataset_path=dataset_path,
        running_mode=running_mode,
    )
    dataset_val = load_and_preprocess_data(
        split="validation",
        batch_size=batch_size,
        shuffle=False,
        image_size=image_size,
        max_examples=max_val_examples,
        dataset_path=dataset_path,
        running_mode=running_mode,
    )

    return dataset_train, dataset_val


def train_unet_model(
        model, 
        dataset_train, 
        dataset_val, 
        epochs : int = 10, 
        model_save_path : str = None, 
        running_mode : str = "debug"
    ) -> tf.keras.callbacks.History:
    
    model_dir = os.path.dirname(model_save_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    log_dir = os.path.join("outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    if importlib.util.find_spec("tensorboard") is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))

    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=epochs,
        callbacks=callbacks,
    )
    model.save(model_save_path)
    
    return history

def save_loss_and_metrics(history: tf.keras.callbacks.History, output_dir: str, run_version: str) -> None:
    """
    Save training loss and metrics to a json file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"training_history_{run_version}.json")

    history_dict = {
        "epoch": list(range(1, len(history.history["loss"]) + 1)),
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "dice_coefficient": history.history.get("dice_coefficient", []),
        "val_dice_coefficient": history.history.get("val_dice_coefficient", []),
        "mean_iou": history.history.get("mean_iou", []),
        "val_mean_iou": history.history.get("val_mean_iou", [])
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, ensure_ascii=False, indent=2)


def visualize_history(history: tf.keras.callbacks.History) -> None:
    """
    Visualize training history (loss and metrics curves).
    """
    import matplotlib.pyplot as plt

    # Plot loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Dice coefficient curves
    if "dice_coefficient" in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history["dice_coefficient"], label="Train Dice Coefficient")
        plt.plot(history.history["val_dice_coefficient"], label="Val Dice Coefficient")
        plt.title("Dice Coefficient Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Coefficient")
        plt.legend()

    plt.tight_layout()
    plt.show()

## ------ Main, debug code, CLI , rapide ------
def main_train_model(running_mode: str = "debug",) -> None:
    """
    Train the U-Net model with a memory-safe preprocessing pipeline.
    """

    # Config ft
    configure_tensorflow()

    # Arguments en dur pour le debug
    model_save_path = os.path.join("outputs", "checkpoints", "unet_debug.keras")
    dataset_path = os.path.join("data", "segment_data")
    batch_size = 4 # 4 pour reduire la memoire RAM
    epochs = 10
    image_size = (256, 256)
    n_filters = 32 # default n_filters : 32 (number of filters in the first encoder block, doubles at each subsequent block)

    exmaples_total_model = 400 # total examples for the model (train + val + test)

    max_train_examples = int(0.8 * exmaples_total_model) # je prends 80% du train dataset (pour training) et 10% pour val dataset (pour validation), phase final : 10% dataset (pour test)
    max_val_examples = int(0.1 * exmaples_total_model) # val 10% du train dataset
    # running_mode only : "debug" ou "full".
    print(f"Running mode: {running_mode}. \n"
          f"Max train examples: {max_train_examples}, \n"
          f"Max val examples: {max_val_examples}, \n"          
          f"Epochs: {epochs}, Model save path: {model_save_path}, \n"
          f"Image size: {image_size}, Batch size: {batch_size}, \n")
    # U-Net modèle
    input_shape = (image_size[0], image_size[1], 3)
    model_unet = compile_unet_model(
        input_shape=input_shape,
        n_filters=n_filters,
        dropout_rate=0.1,
        learning_rate=1e-4,
        weight_decay=1e-4
    )

    # datasets
    dataset_train, dataset_val = train_val_dataset_preprocessed(
        dataset_path=dataset_path,
        image_size=image_size,
        batch_size=batch_size,
        max_train_examples=max_train_examples,
        max_val_examples=max_val_examples,
        running_mode=running_mode,
    )

    # train modèle
    history = train_unet_model(
        model=model_unet,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        epochs=epochs,
        model_save_path=model_save_path,
        running_mode=running_mode,
    )
    # Save training history to a text file
    run_version = f"v_{time.strftime('%M%S')}" # v_minutesseconds like v_2233, minutes : 22, seconds: 33
    save_loss_and_metrics(history, output_dir=os.path.join("outputs", "metrics"), run_version=run_version)

    # visualize_history(history)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training the U-Net model.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train U-Net on CATMuS line segmentation.")
    parser.add_argument("--running-mode", choices=["debug", "full"], default="debug")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main_train_model(
        running_mode=args.running_mode,
    )

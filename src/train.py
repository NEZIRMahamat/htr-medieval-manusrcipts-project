from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
import tensorflow as tf

try:
    from .model import compile_unet_model
    from .utils import (
        DEFAULT_IMAGE_SIZE,
        load_segmentation_dataset,
        preprocess_image_and_mask,
        select_split_subset,
    )
except ImportError:
    from model import compile_unet_model
    from utils import (
        DEFAULT_IMAGE_SIZE,
        load_segmentation_dataset,
        preprocess_image_and_mask,
        select_split_subset,
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

    This is useful only for tiny debug subsets. Prefer load_and_preprocess_data for training.
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
) -> tf.data.Dataset:
    """
    Stream preprocessed CATMuS examples as TensorFlow batches.

    Output shapes:
        image: (height, width, 3), float32 in [0, 1]
        mask:  (height, width, 1), float32 in {0, 1}
    """
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

    def generator():
        for example in split_dataset:
            yield preprocess_image_and_mask(example, target_size=image_size)

    dataset_tf = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset_tf = dataset_tf.apply(tf.data.experimental.assert_cardinality(cardinality))

    if shuffle:
        shuffle_buffer = min(cardinality, max_examples if max_examples is not None else 256)
        dataset_tf = dataset_tf.shuffle(
            buffer_size=max(1, shuffle_buffer),
            reshuffle_each_iteration=True,
        )

    return dataset_tf.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_unet_model(
    model_save_path: str,
    batch_size: int = 1,
    epochs: int = 2,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    n_filters: int = 8,
    max_train_examples: int | None = 8,
    max_val_examples: int | None = 2,
    dataset_path: str | None = None,
    log_dir: str = os.path.join("outputs", "logs"),
) -> tf.keras.callbacks.History:
    """
    Train the U-Net model with a memory-safe preprocessing pipeline.
    """
    configure_tensorflow()

    input_shape = (image_size[0], image_size[1], 3)
    model_unet = compile_unet_model(
        input_shape=input_shape,
        n_filters=n_filters,
        dropout_rate=0.1,
        learning_rate=1e-4,
        weight_decay=1e-5,
    )

    model_dir = os.path.dirname(model_save_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

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

    dataset_train = load_and_preprocess_data(
        split="train",
        batch_size=batch_size,
        shuffle=True,
        image_size=image_size,
        max_examples=max_train_examples,
        dataset_path=dataset_path,
    )
    dataset_val = load_and_preprocess_data(
        split="validation",
        batch_size=batch_size,
        shuffle=False,
        image_size=image_size,
        max_examples=max_val_examples,
        dataset_path=dataset_path,
    )

    history = model_unet.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=epochs,
        callbacks=callbacks,
    )

    model_unet.save(model_save_path)
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net on CATMuS line segmentation.")
    parser.add_argument("--model-save-path", default=os.path.join("outputs", "checkpoints", "unet_debug.keras"))
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n-filters", type=int, default=8)
    parser.add_argument("--max-train-examples", type=int, default=8, help="0 means full train split.")
    parser.add_argument("--max-val-examples", type=int, default=2, help="0 means full validation split.")
    parser.add_argument("--log-dir", default=os.path.join("outputs", "logs"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_size = (args.image_size, args.image_size)

    train_unet_model(
        model_save_path=args.model_save_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        image_size=image_size,
        n_filters=args.n_filters,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
        dataset_path=args.dataset_path,
        log_dir=args.log_dir,
    )

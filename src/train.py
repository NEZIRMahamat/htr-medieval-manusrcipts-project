import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from model import compile_unet_model
from utils import preprocess_images_and_masks



def make_tf_dataset(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    batch_size: int = 8,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from images and masks.
    Args:
        images (list[np.ndarray]): List of preprocessed images.
        masks (list[np.ndarray]): List of preprocessed masks.
        batch_size (int): Batch size for the dataset.
        shuffle (bool): Whether to shuffle the dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size)
    return dataset


def load_and_preprocess_data(split: str = "train", batch_size: int = 8, shuffle: bool = True) -> tf.data.Dataset:
    """
    Load and preprocess images and masks from a HuggingFace dataset saved on disk.
    Args:
        split (str): Dataset split to load ("train", "validation", "test").
    Returns:
        tf.data.Dataset: TensorFlow dataset of preprocessed images and masks.
    """
    images, masks = preprocess_images_and_masks(split=split)
    dataset_tf = make_tf_dataset(images, masks, batch_size=batch_size, shuffle=shuffle)
    return dataset_tf

# Entrainement du modèle U-Net avec les données prétraitées
def train_unet_model(model_save_path: str, batch_size: int = 8, epochs: int = 30) -> None:
    input_shape = (256, 256, 3)
    n_filters = 32
    model_unet = compile_unet_model(
        input_shape=input_shape, n_filters=n_filters, 
        dropout_rate=0.1, learning_rate=1e-4, weight_decay=1e-5
    )

    # Callbacks for early stopping, model checkpointing, and TensorBoard logging
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir="logs"),
    ]

    dataset_train = load_and_preprocess_data(split="train", batch_size=batch_size, shuffle=True)
    dataset_val = load_and_preprocess_data(split="validation", batch_size=batch_size, shuffle=False)

    history = model_unet.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Save the final model after training
    model_unet.save(model_save_path)

## ------ Main, test rapide et debugging développement.


if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")

    train_unet_model(
        model_save_path=os.path.join("outputs", "unet_model.h5"),
        batch_size=8,
        epochs=30
    )

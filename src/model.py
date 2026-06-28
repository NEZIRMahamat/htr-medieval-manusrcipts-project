"""
Model definition for U-Net architecture using TensorFlow and Keras.
This module defines the U-Net model architecture, including the encoder and decoder blocks,
as well as the loss functions and metrics used for training and evaluation.
Objectives:
- Define the U-Net architecture for image segmentation tasks.
- Implement encoder and decoder blocks with convolutional layers, batch normalization, and dropout.
- Provide loss functions (Dice loss, combined loss) and metrics (Dice coefficient, Mean IoU) for model evaluation.
Author: Nezir Abakar
"""

from tensorflow.keras.layers import ( # Type : ignore
    BatchNormalization,
    Dropout,
    Input, 
    Conv2D, 
    MaxPooling2D,
    Conv2DTranspose, 
    concatenate 
    )
from tensorflow.keras.models import Model # Type : ignore
from tensorflow.keras.optimizers import AdamW # Type : ignore
import tensorflow as tf


# Double convolutional block
def double_conv_block(input_tensor, n_filters, kernelsize=3, activation='relu'):
    """
    Double convolutional block with Conv2D, BatchNormalization, and ReLU activation.
    """
    x = Conv2D(n_filters, kernelsize, padding="same", activation=activation)(input_tensor)
    x = Conv2D(n_filters, kernelsize, padding="same", activation=activation)(x)
    x = BatchNormalization()(x)
    return x


# Encodeur
def build_encoder(inputs, n_filters=32):
    skip = double_conv_block(inputs, n_filters)
    x = MaxPooling2D((2, 2))(skip)
    return skip, x



# Decodeur
def build_decoder(encoded_inputs, skip, n_filters=32):
    x_encoded = Conv2DTranspose(n_filters, kernel_size=3, strides=(2, 2), padding="same")(encoded_inputs)   
    x_concat = concatenate([x_encoded, skip])
    x = double_conv_block(x_concat, n_filters)
    return x


# Build U-Net model
def build_unet(input_shape, n_filters=32, dropout_rate=0.1):
    inputs = Input(shape=input_shape)

    # encoder
    skip1, x1 = build_encoder(inputs, n_filters) # n_filters : 32
    x1 = Dropout(dropout_rate)(x1)  # Dropout layer after the first encoder block
    
    skip2, x2 = build_encoder(x1, n_filters * 2) # n_filters * 2 : 64
    x2 = Dropout(dropout_rate)(x2)  # Dropout layer after the second encoder block
    
    skip3, x3 = build_encoder(x2, n_filters * 4) # n_filters * 4 : 128
    x3 = Dropout(dropout_rate)(x3)  # Dropout layer after the third encoder block
    
    skip4, x4 = build_encoder(x3, n_filters * 8) # n_filters * 8 : 256
    x4 = Dropout(dropout_rate)(x4)  # Dropout layer after the fourth encoder block

    # bottleneck
    x_bottleneck = double_conv_block(x4, n_filters * 16)
    
    # decoder
    y1 = build_decoder(x_bottleneck, skip4, n_filters * 8) # n_filters * 8 : 256 (decoder 1)
    y2 = build_decoder(y1, skip3, n_filters * 4) # n_filters * 4 : 128 (decoder 2)
    y3 = build_decoder(y2, skip2, n_filters * 2) # n_filters * 2 : 64 (decoder 3)
    y4 = build_decoder(y3, skip1, n_filters) # n_filters : 32 (decoder 4)

    # model
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(y4)

    model = Model(inputs=inputs, outputs=outputs)

    return model


# Metrics and loss functions

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient metric for evaluating segmentation performance.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    """
    Dice loss function for training segmentation models.
    """
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred, alpha=0.5):
    """
    Combined loss function: Dice loss + Binary Crossentropy loss.
    """
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return alpha * bce + (1 - alpha) * d_loss


def mean_iou(y_true, y_pred):
    """
    Mean Intersection over Union (IoU) metric for evaluating segmentation performance.
    """
    epsilon = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    return (intersection + epsilon) / (union + epsilon)


def compile_unet_model(
        input_shape, 
        n_filters=32, 
        dropout_rate=0.1, 
        learning_rate=1e-4, 
        weight_decay=1e-4):
    """
    Compile the U-Net model with the specified parameters.
    """
    model = build_unet(input_shape, n_filters, dropout_rate)
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)  # Using AdamW optimizer with weight decay
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=[dice_coefficient, mean_iou])
    return model



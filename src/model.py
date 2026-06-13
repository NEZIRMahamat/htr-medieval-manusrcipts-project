from tensorflow.keras.layers import (
    Input, # Input layer (entrée du modèle, spécifie la forme des données d'entrée shape)
    Conv2D, # Convolution 2D
    MaxPooling2D, # Max Pooling 2D
    UpSampling2D, # UpSampling 2D
    Conv2DTranspose, # Convolution Transpose 2D (ou Deconvolution, decoder)
    concatenate # Concatenate layers
    )
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
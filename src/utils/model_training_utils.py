import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Dropout, GRU, LSTM, Bidirectional, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1, l2
from keras.activations import relu, softmax
from keras.losses import sparse_categorical_crossentropy

def build_convolutional_model(seq_len, feature_dim = 1, no_classes=9) -> keras.Sequential:
    model = keras.Sequential(
    [
        layers.Input(shape=(seq_len, feature_dim)),
        layers.Conv1D(
            filters=20, kernel_size=8, padding="same", strides=2, activation="relu", kernel_initializer="uniform"
        ),
        # layers.Dropout(rate=0.1),
        layers.MaxPool1D(20, padding="same", strides = 1),
        layers.Conv1D(
            filters=12, kernel_size=6, padding="same", strides=2, activation="relu", kernel_initializer="uniform"
        ),
        layers.MaxPool1D(8, padding="same", strides = 1),
        layers.Flatten(),
        # layers.Dense(100, activation = relu, kernel_initializer="uniform"),
        layers.Dense(50, activation = relu, kernel_initializer="uniform"),
        layers.Dense(25, activation = relu, kernel_initializer="uniform"),
        layers.Dense(10, activation = relu, kernel_initializer="uniform"),
        layers.Dense(9, activation = softmax)
        # layers.Dense(1, activation = relu, kernel_initializer="uniform"),
    ]
    )
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=1e-3), loss=sparse_categorical_crossentropy)
    return model


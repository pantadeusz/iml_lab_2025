import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


def get_data():
    (x_train, _), (x_test, _) = fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # print(x_train.shape)
    # print(x_test.shape)

    return x_train, x_test


class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=shape),
                layers.Conv2D(16, (3, 3), strides=2, activation="relu", padding="same"),
                layers.Conv2D(32, (3, 3), strides=1, activation="relu", padding="same"),
                layers.Flatten(),
                layers.Dense(latent_dim, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(14 * 14 * 32, activation="relu"),
                layers.Reshape((14, 14, 32)),
                layers.Conv2DTranspose(
                    32, kernel_size=3, strides=1, activation="relu", padding="same"
                ),
                layers.Conv2DTranspose(
                    16, kernel_size=3, strides=2, activation="relu", padding="same"
                ),
                layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def create_autoencoder(shape, latent_dim=64):
    autoencoder = Autoencoder(latent_dim, shape)
    autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

    return autoencoder


def data_augmention(train, test):
    rotator = layers.RandomRotation(factor=0.2, fill_mode="constant", fill_value=0.0)

    train_rotated = rotator(train, training=True)
    test_rotated = rotator(test, training=True)

    return train_rotated, test_rotated


def save_both_models(autoencoder: Autoencoder, name):
    path = os.path.join("models", name)
    os.makedirs(path, exist_ok=True)

    autoencoder.encoder.save(f"models/{name}/encoder.keras")
    autoencoder.decoder.save(f"models/{name}/decoder.keras")


def main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = None

    x_train, x_test = get_data()
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    shape = x_test.shape[1:]

    x_train_rotated, x_test_rotated = data_augmention(x_train, x_test)

    autoencoder = create_autoencoder(shape)

    autoencoder.fit(
        x_train_rotated,
        x_train,
        epochs=10,
        shuffle=True,
        validation_data=(x_test, x_test),
    )

    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test_rotated[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("rotation_conv.png")

    if name is not None:
        save_both_models(autoencoder, name)


if __name__ == "__main__":
    main()

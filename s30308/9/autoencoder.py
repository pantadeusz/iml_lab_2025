import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


def load_data():
    (x_train, _), (x_test, _) = fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    return x_train, x_test


def rotate_ds(ds):
    # Dodajemy wymiar kanału (28, 28) -> (28, 28, 1)
    # Bez tego RandomRotation bierze ostatni wymiar jako ilość kanałów przez co wynik wychodzi zniekształcony
    ds = tf.expand_dims(ds, -1)

    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(
            factor=(-0.1, 0.1),
            fill_mode='constant',
            fill_value=0.0,  # Wypełniamy czarnym kolorem (0.0)
            interpolation='bilinear'
        )
    ])

    augmented = data_augmentation(ds)

    # Usuwamy wymiar kanału, aby wrócić do formatu (28, 28)
    return tf.squeeze(augmented, axis=-1).numpy()

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(x_train, x_test):
    shape = x_test.shape[1:]
    latent_dim = 64
    autoencoder = Autoencoder(latent_dim, shape)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    x_train_rotated = rotate_ds(x_train)
    x_test_rotated = rotate_ds(x_test)

    autoencoder.fit(x_train_rotated, x_train,
                    epochs=10,
                    shuffle=True,
                    validation_data=(x_test, x_test))


    encoded_imgs = autoencoder.encoder(x_test_rotated).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    show_reconstruction(x_train_rotated, decoded_imgs)

    return encoded_imgs, decoded_imgs

def show_reconstruction(output_imgs, result_imgs):
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(output_imgs[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(result_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    x_train, x_test = load_data()

    encoded_imgs, decoded_imgs = train_autoencoder(x_train, x_test)
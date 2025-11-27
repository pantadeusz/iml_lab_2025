import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import argparse

def init_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-r', '--rotation', default=0.1)
    return argparser.parse_args()

def get_data():
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    return x_train, x_test

def normalize_data(x_train, x_test):
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    return x_train, x_test

def get_augmented_data(x_train, rotation_factor=0.1):
    x_train = np.expand_dims(x_train, axis=-1)

    augmentation_pipeline = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(rotation_factor, fill_mode='constant')
    ])

    x_augmented = augmentation_pipeline(x_train).numpy()

    plt.imshow(x_augmented[0].squeeze(), cmap='gray')
    plt.show()

    x_augmented = np.squeeze(x_augmented)
    return x_augmented

def create_and_train_autoencoder(x_train, x_test):
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

    shape = x_test.shape[1:]
    latent_dim = 64
    autoencoder = Autoencoder(latent_dim, shape)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    return autoencoder

def show_augmented_and_after_autoencoder(x_train, autoencoder):
    results = autoencoder(x_train).numpy()

    n_cols = 8
    fig, axes = plt.subplots(2, n_cols, figsize=(16, 4))

    for col_i in range(n_cols):
        axes[0][col_i].imshow(x_train[col_i], cmap='plasma')
        axes[0][col_i].axis('off')

        axes[1][col_i].imshow(results[col_i], cmap='plasma')
        axes[1][col_i].axis('off')

    axes[0][0].set_ylabel("Original", fontsize=14)
    axes[1][0].set_ylabel("Reconstructed", fontsize=14)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = init_argparser()
    rotation = args.rotation
    
    x_train, x_test = get_data()
    x_train, x_test = normalize_data(x_train, x_test)
    autoencoder = create_and_train_autoencoder(x_train, x_test)
    x_train_augmented = get_augmented_data(x_train, rotation_factor=rotation)

    show_augmented_and_after_autoencoder(x_train_augmented, autoencoder)
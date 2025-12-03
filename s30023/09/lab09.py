import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim

    self.encoder = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='elu', padding='same', strides=2), # 28x28 -> 14x14 [x32]
        layers.Conv2D(64, (3, 3), activation='elu', padding='same', strides=2), # 14x14 -> 7x7 [x64]
        layers.Flatten(), #7x7x64
        layers.Dense(latent_dim)
    ])

    self.decoder = tf.keras.Sequential([
        layers.Dense(7*7*64, activation='elu'),
        layers.Reshape((7, 7, 64)),
        layers.Conv2DTranspose(64, (3, 3), activation='elu', padding='same', strides=2), #7x7 -> 14x14 [x64]
        layers.Conv2DTranspose(32, (3, 3), activation='elu', padding='same', strides=2), #14x14 -> 28x28 [x32]
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same') # -> 28x28 [x1]
    ])

    # https://www.tensorflow.org/tutorials/generative/cvae

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.2, fill_mode='constant', fill_value=0.0),
])

def load_mnist():
    (x_train, _), (x_test, _) = fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return x_train, x_test

def augment_image(image):
    return data_augmentation(image)

def train(x_train, x_test, epochs=10):
    x_train_augmented = augment_image(x_train)
    x_test_augmented = augment_image(x_test)

    print(f'x_train shape: {x_train.shape}')

    autoencoder = Autoencoder(latent_dim=64)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.fit(x_train_augmented, x_train,
                    epochs=epochs,
                    shuffle=True,
                    validation_data=(x_test_augmented, x_test))

    return autoencoder, x_test_augmented


def compare(autoencoder, x_test_rotated, x_test_original):
    decoded_imgs = autoencoder.call(x_test_rotated).numpy()

    n = 10
    plt.figure(figsize=(20, 6))  # Zwiększona wysokość
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test_rotated[i])
        plt.title("input (rotated)")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("output (fixed)")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(x_test_original[i])
        plt.title("target (original)")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()

def main():
    x_train, x_test = load_mnist()

    autoencoder, x_test_augmented = train(x_train, x_test, epochs=10)

    compare(autoencoder, x_test_augmented, x_test)

    print(f'Encoder summary: \n{autoencoder.encoder.summary()}')
    print(f'Decoder summary: \n{autoencoder.decoder.summary()}')

    autoencoder.encoder.save('encoder_model.keras')
    autoencoder.decoder.save('decoder_model.keras')

if __name__ == '__main__':
    main()
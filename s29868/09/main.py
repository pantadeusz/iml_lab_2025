import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


augmentator = tf.keras.Sequential([
        layers.RandomRotation(0.25, fill_mode='nearest')
    ])


def load_data():
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, x_test


def augment_data(data, augmentator_model):
    return augmentator_model(data)


class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape

        self.encoder = tf.keras.Sequential([
            layers.InputLayer(shape=shape),
            layers.Conv2D(32, (3, 3), activation='tanh', padding='same', strides=2),
            layers.Conv2D(64, (3, 3), activation='tanh', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(latent_dim),
        ])
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(shape=(latent_dim,)),
            layers.Dense(7 * 7 * 64, activation='tanh'),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(32, (3, 3), activation='tanh', padding='same', strides=2),
            layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same', strides=2),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_model(model, x_train_in, x_train_target, x_test_in, x_test_target):
    model.compile(optimizer='adam', loss=losses.MeanSquaredError())
    model.fit(x_train_in, x_train_target,
              epochs=5,
              shuffle=True,
              validation_data=(x_test_in, x_test_target))


def get_predictions(model, x_input):
    encoded_imgs = model.encoder(x_input).numpy()
    decoded_imgs = model.decoder(encoded_imgs).numpy()
    return decoded_imgs


def plot_results(x_input, x_decoded):
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(np.squeeze(x_input[i]), vmin=0, vmax=1, cmap='gray')
        plt.title("original input")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(np.squeeze(x_decoded[i]), vmin=0, vmax=1, cmap='gray')
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def save_model_parts(model):
    model.encoder.save('encoder.h5')
    model.decoder.save('decoder.h5')


def main():
    x_train, x_test = load_data()
    x_train_aug = augment_data(x_train, augmentator)
    x_test_aug = augment_data(x_test, augmentator)

    shape = x_test.shape[1:]
    latent_dim = 64

    autoencoder = Autoencoder(latent_dim, shape)

    train_model(autoencoder, x_train_aug, x_train, x_test_aug, x_test)

    decoded_imgs = get_predictions(autoencoder, x_test_aug)

    plot_results(x_test_aug, decoded_imgs)

    save_model_parts(autoencoder)


if __name__ == '__main__':
    main()
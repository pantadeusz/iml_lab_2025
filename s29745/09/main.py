import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


def get_data():
    (x_train, _), (x_test, _) = fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    augmentation = tf.keras.Sequential(
        [layers.RandomRotation(factor=0.1, fill_mode="constant", fill_value=0.0)]
    )

    x_train_rotated = augmentation(x_train).numpy()
    x_test_rotated = augmentation(x_test).numpy()

    return x_train, x_train_rotated, x_test, x_test_rotated, x_train.shape[1:]


class DenseAutoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(DenseAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(latent_dim, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(tf.math.reduce_prod(shape).numpy(), activation="sigmoid"),
                layers.Reshape(shape),
            ]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "shape": self.shape,
            }
        )
        return config

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvAutoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape

        self.encoder = tf.keras.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Flatten(),
                layers.Dense(latent_dim, activation="relu"),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense((shape[0] // 2) * (shape[1] // 2) * 32, activation="relu"),
                layers.Reshape((shape[0] // 2, shape[1] // 2, 32)),
                layers.Conv2DTranspose(
                    32, (3, 3), strides=2, activation="relu", padding="same"
                ),
                layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "shape": self.shape,
            }
        )
        return config

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_save_plot(model, name, x_train, x_train_rot, x_test, x_test_rot):
    model.compile(optimizer="adam", loss=losses.MeanSquaredError())

    print(f"Trenowanie modelu: {name}...")
    model.fit(
        x_train_rot,
        x_train,
        epochs=10,
        shuffle=True,
        validation_data=(x_test_rot, x_test),
    )

    encoder_filename = f"{name}_encoder.keras"
    decoder_filename = f"{name}_decoder.keras"

    model.encoder.save(encoder_filename)
    model.decoder.save(decoder_filename)
    print(f"Zapisano: {encoder_filename} oraz {decoder_filename}")

    n = 10
    x_test_subset = x_test_rot[:n]

    encoded_imgs = model.encoder(x_test_subset).numpy()
    decoded_imgs = model.decoder(encoded_imgs).numpy()

    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test_subset[i].squeeze())
        plt.title("input")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].squeeze())
        plt.title(name)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(f"results_{name}.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    x_train, x_train_rot, x_test, x_test_rot, input_shape = get_data()
    latent_dim = 64

    dense_ae = DenseAutoencoder(latent_dim, input_shape)
    train_save_plot(
        dense_ae, "dense_autoencoder", x_train, x_train_rot, x_test, x_test_rot
    )

    conv_ae = ConvAutoencoder(latent_dim, input_shape)
    train_save_plot(
        conv_ae, "conv_autoencoder", x_train, x_train_rot, x_test, x_test_rot
    )

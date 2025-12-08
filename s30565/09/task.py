import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

(x_train, _), (x_test, _) = fashion_mnist.load_data()


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)


rotation_layer = layers.RandomRotation(factor=0.15)  

x_train_rotated = rotation_layer(x_train, training=True)
x_test_rotated = rotation_layer(x_test, training=True)

latent_dim = 64


class Autoencoder(Model):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        encoder_inputs = layers.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(encoder_inputs)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 14x14
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 7x7
        x = layers.Flatten()(x)
        latent = layers.Dense(latent_dim, activation="relu")(x)
        self.encoder = Model(encoder_inputs, latent, name="encoder")

        decoder_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(decoder_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(
            64, (3, 3), strides=2, padding="same", activation="relu"
        )(x)  
        x = layers.Conv2DTranspose(
            32, (3, 3), strides=2, padding="same", activation="relu"
        )(x)  
        decoder_outputs = layers.Conv2D(
            1, (3, 3), activation="sigmoid", padding="same"
        )(x)
        self.decoder = Model(decoder_inputs, decoder_outputs, name="decoder")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

autoencoder.summary()

history = autoencoder.fit(
    x_train_rotated,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_rotated, x_test),
)


autoencoder.encoder.save("encoder.h5")
autoencoder.decoder.save("decoder.h5")
autoencoder.save("autoencoder_full.h5")

encoded_imgs = autoencoder.encoder(x_test_rotated).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_rotated[i].squeeze(), cmap="gray")
    plt.title("obrócony")
    plt.axis("off")

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].squeeze(), cmap="gray")
    plt.title("prostowany")
    plt.axis("off")

plt.tight_layout()
plt.show()
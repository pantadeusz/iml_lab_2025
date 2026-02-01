import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 64
EPOCHS = 10
LATENT_DIM = 64

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

rotator = tf.keras.layers.RandomRotation(0.1)


def create_dataset(images, is_train=True):
    ds = tf.data.Dataset.from_tensor_slices(images)
    if is_train:
        ds = ds.shuffle(10000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda img: (rotator(img), img))
    return ds


train_ds = create_dataset(x_train, is_train=True)
test_ds = create_dataset(x_test, is_train=False)


class ConvAutoencoder(Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = ConvAutoencoder()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(train_ds, epochs=EPOCHS, validation_data=test_ds)

for rotated_imgs, original_imgs in test_ds.take(1):
    fixed_imgs = autoencoder(rotated_imgs)

    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(tf.squeeze(rotated_imgs[i]), cmap='gray')
        plt.title("Wejście (Krzywe)")
        plt.axis('off')

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(tf.squeeze(fixed_imgs[i]), cmap='gray')
        plt.title("Wyjście (Autoencoder)")
        plt.axis('off')

        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(tf.squeeze(original_imgs[i]), cmap='gray')
        plt.title("Cel (Prosty)")
        plt.axis('off')

    plt.savefig('zadanie3_wynik.png')

autoencoder.encoder.save('encoder_model.keras')
autoencoder.decoder.save('decoder_model.keras')
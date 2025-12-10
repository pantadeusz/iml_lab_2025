import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import models


def load_models():
    encoder = models.load_model('encoder.h5')
    decoder = models.load_model('decoder.h5')
    return encoder, decoder


def load_images():
    (_, _), (x_test, _) = fashion_mnist.load_data()

    img = x_test[np.random.choice(len(x_test))]
    img = img.astype('float32') / 255.
    img = tf.expand_dims(img, -1)
    img = tf.expand_dims(img, 0)

    augmentator = tf.keras.layers.RandomRotation(0.25, fill_mode='nearest')
    augmented_img = augmentator(img)

    return img, augmented_img


def straighten_images(encoder, decoder, rotated_img):
    latent_vector = encoder(rotated_img)
    decoded_img = decoder(latent_vector)
    return decoded_img, latent_vector


def plot_result(decoded_img, encoded_img):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title('Rotated Image')
    plt.imshow(np.squeeze(encoded_img), vmin=0, vmax=1, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Decoded Image')
    plt.imshow(np.squeeze(decoded_img), vmin=0, vmax=1, cmap='gray')
    plt.show()


def main():
    encoder, decoder = load_models()
    img, augmented_img = load_images()
    decoded_img, latent_vector = straighten_images(encoder, decoder, augmented_img)
    print(f" Latent vector: START \n{latent_vector}\nEND")
    plot_result(decoded_img, augmented_img)


if __name__ == '__main__':
    main()

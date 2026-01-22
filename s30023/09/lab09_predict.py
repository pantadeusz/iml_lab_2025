# predict_model.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import cv2


def load_trained_models():
    encoder = tf.keras.models.load_model('encoder_model.keras')
    decoder = tf.keras.models.load_model('decoder_model.keras')
    return encoder, decoder


def get_sample_image():
    # (_, _), (x_test, _) = fashion_mnist.load_data()
    #
    # index_of_random_picture = np.random.randint(0, len(x_test))
    # image = x_test[index_of_random_picture]

    image = cv2.imread('Group.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype('float32') / 255.
    image = image[..., tf.newaxis]
    image_batch = tf.expand_dims(image, 0)

    return image_batch, image


def process_image(encoder, decoder, input_image):
    latent_vector = encoder.predict(input_image)

    decoded_image = decoder.predict(latent_vector)

    return latent_vector, decoded_image

def compare_image(output_image, input_image):
    output_image = output_image[0, :, :, 0]
    input_image = input_image[0, :, :, 0]

    # plt.imsave("result.png", output_image, cmap='gray')

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("input (rotated)")
    plt.imshow(input_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("output (fixed)")
    plt.imshow(output_image, cmap='gray')
    plt.axis('off')

    plt.show()

def main():
    encoder, decoder = load_trained_models()

    input_img, original_img = get_sample_image()

    latent_vector, output_img = process_image(encoder, decoder, input_img)

    print("\n--- Latent Vector ---")
    print(f"Shape of latent vector: {latent_vector.shape}")
    print(f"Values: {latent_vector}")

    compare_image(output_img, input_img)


if __name__ == '__main__':
    main()
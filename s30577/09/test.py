import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os


IMAGE_FILENAME = 'photo.jpg'


def prepare_image(filename):
    img = image.load_img(filename, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = 1.0 - img_array
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch, img


encoder = tf.keras.models.load_model('encoder_model.keras')
decoder = tf.keras.models.load_model('decoder_model.keras')

input_batch, original_pil = prepare_image(IMAGE_FILENAME)

rotator = tf.keras.layers.RandomRotation(0.1)
rotated_batch = rotator(input_batch, training=True)

latent_vector = encoder.predict(rotated_batch)

decoded_output = decoder.predict(latent_vector)

print(f"Wektor ukryty (kształt): {latent_vector.shape}")

plt.figure(figsize=(12, 4))

ax = plt.subplot(1, 4, 1)
plt.imshow(image.img_to_array(original_pil).squeeze(), cmap='gray')
plt.title("Oryginał (28x28)")
plt.axis('off')

ax = plt.subplot(1, 4, 2)
plt.imshow(tf.squeeze(rotated_batch), cmap='gray')
plt.title("Wejście sieci\n(Negatyw + Obrót)")
plt.axis('off')

ax = plt.subplot(1, 4, 3)
feature_map = tf.reduce_mean(latent_vector[0], axis=-1)
plt.imshow(feature_map, cmap='viridis')
plt.title("Wektor ukryty\n(Latent)")
plt.axis('off')

ax = plt.subplot(1, 4, 4)
plt.imshow(tf.squeeze(decoded_output), cmap='gray')
plt.title("Wynik")
plt.axis('off')

plt.savefig('wynik.png')

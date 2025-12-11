from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
encoder_path = os.path.abspath("encoder.keras")
decoder_path = os.path.abspath("decoder.keras")

print("FULL PATH:", encoder_path)

def load_image(path, target_size=(28,28)):
    img = Image.open(path).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def save_image(img_array, path):
    img_array = np.squeeze(img_array)
    img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(path)

if __name__ == "__main__":
    encoder_path = "encoder.keras"
    decoder_path = "decoder.keras"

    encoder = keras.models.load_model(encoder_path)
    decoder = keras.models.load_model(decoder_path)

    input_path = "mist.png"
    output_path = "straightened.png"

    img = load_image(input_path)

    latent_vector = encoder(img)
    print("Latent vector:", latent_vector.numpy())

    reconstructed = decoder(latent_vector)

    save_image(reconstructed, output_path)
    print(f"Zapisano prostowany obraz do {output_path}")

    plt.figure(figsize=(4,2))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(np.squeeze(img), cmap='gray')
    plt.subplot(1,2,2)
    plt.title("Reconstructed")
    plt.imshow(np.squeeze(reconstructed), cmap='gray')
    plt.show()
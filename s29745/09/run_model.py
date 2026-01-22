import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_process_image(image_path):
    if not os.path.exists(image_path):
        print(f"Plik '{image_path}' nie istnieje.")
        return None

    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_ready = np.expand_dims(img_array, 0)
    return img_ready

def main():
    if len(sys.argv) < 4:
        print("Uzycie: python skrypt.py <sciezka_enkodera> <sciezka_dekodera> <sciezka_obrazka>")
        return

    encoder_path = sys.argv[1]
    decoder_path = sys.argv[2]
    image_path = sys.argv[3]

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print("Nie znaleziono pliku enkodera lub dekodera.")
        return

    try:
        encoder = tf.keras.models.load_model(encoder_path)
        decoder = tf.keras.models.load_model(decoder_path)
    except Exception as e:
        print(f"Blad podczas ladowania modeli: {e}")
        return

    input_img = load_and_process_image(image_path)
    if input_img is None:
        return

    latent_vector = encoder.predict(input_img, verbose=0)
    output_img = decoder.predict(latent_vector, verbose=0)

    print("="*40)
    print(f"Obraz: {image_path}")
    print(f"Encoder: {encoder_path}")
    print("WEKTOR UKRYTY:")
    print(latent_vector[0])
    print("="*40)

    out_name = f"out_{os.path.basename(image_path)}"
    img_to_save = output_img[0, :, :, 0]
    
    plt.imsave(out_name, img_to_save, cmap='gray')
    print(f"Zapisano wynik jako: {out_name}")

if __name__ == "__main__":
    main()
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model


def load_autoencoder(encoder_path='encoder_model.keras', decoder_path='decoder_model.keras'):
    print(f"Ładowanie enkodera z: {encoder_path}")
    encoder = load_model(encoder_path)
    print(f"Ładowanie dekodera z: {decoder_path}")
    decoder = load_model(decoder_path)
    return encoder, decoder


def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype='float32')
    img_array = img_array / 255.0
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array

    return img_array


def straighten_image(encoder, decoder, image):
    input_img = image[np.newaxis, ..., np.newaxis]
    latent_vector = encoder.predict(input_img, verbose=0)
    output_img = decoder.predict(latent_vector, verbose=0)
    return output_img[0].squeeze(), latent_vector[0]


def main():
    parser = argparse.ArgumentParser(description='Prostowanie obrazków Fashion MNIST')
    parser.add_argument('input', type=str, help='Ścieżka do obrazka wejściowego')
    parser.add_argument('-o', '--output', type=str, default='wyprostowany.png',
                        help='Ścieżka do obrazka wyjściowego (domyślnie: wyprostowany.png)')
    parser.add_argument('--encoder', type=str, default='encoder_model.keras',
                        help='Ścieżka do modelu enkodera')
    parser.add_argument('--decoder', type=str, default='decoder_model.keras',
                        help='Ścieżka do modelu dekodera')

    args = parser.parse_args()

    # Ładowanie modeli
    try:
        encoder, decoder = load_autoencoder(args.encoder, args.decoder)
    except Exception as e:
        print(f"Błąd ładowania modeli: {e}")
        print("Upewnij się, że najpierw uruchomiłeś autoencoder_rotation.py")
        return

    # Wczytanie i przetworzenie obrazka
    print(f"\nWczytywanie obrazka: {args.input}")
    try:
        input_image = preprocess_image(args.input)
    except Exception as e:
        print(f"Błąd wczytywania obrazka: {e}")
        return

    # Prostowanie
    print("Prostowanie obrazka...")
    output_image, latent_vector = straighten_image(encoder, decoder, input_image)

    # Zapisanie wyniku - porównanie wejścia i wyjścia
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title("Wejście (krzywe)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_image, cmap='gray')
    plt.title("Wyjście (proste)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Zapisano wynik do: {args.output}")
    plt.close()

    # Zapisz też sam wyprostowany obrazek
    output_only_path = args.output.replace('.png', '_tylko_wynik.png')
    plt.figure(figsize=(3, 3))
    plt.imshow(output_image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_only_path, dpi=150, bbox_inches='tight', pad_inches=0)
    print(f"Zapisano sam wynik do: {output_only_path}")
    plt.close()

    # Wyświetlenie wektora ukrytego
    print("\n" + "=" * 60)
    print("WEKTOR UKRYTY (LATENT)")
    print("=" * 60)
    print(f"Wymiar: {len(latent_vector)}")
    print(f"Wartości:")
    print(latent_vector)
    print("=" * 60)


if __name__ == "__main__":
    main()

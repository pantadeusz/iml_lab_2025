import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def load_models():
    """Ładuje zapisane wcześniej modele enkodera i dekodera."""
    encoder = tf.keras.models.load_model("encoder.h5")
    decoder = tf.keras.models.load_model("decoder.h5")
    return encoder, decoder


def get_example_image(index: int = 0):
    """
    Dla prostoty bierzemy obrazek z testowego zbioru Fashion MNIST.
    Używamy tej samej normalizacji co przy trenowaniu.
    """
    (_, _), (x_test, _) = fashion_mnist.load_data()

    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)  # (N, 28, 28, 1)

    index = int(index)
    if index < 0 or index >= len(x_test):
        raise ValueError(f"Index poza zakresem, musi być w [0, {len(x_test) - 1}]")

    img = x_test[index]  # (28, 28, 1)
    return img


def main():
    # Opcjonalny argument z linii komend: indeks obrazka z test setu
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    else:
        idx = 0

    print(f"Używam obrazka o indeksie: {idx}")

    encoder, decoder = load_models()

    # Pobieramy jeden obrazek
    img = get_example_image(idx)  # (28, 28, 1)

    # Dodajemy wymiar batcha: (1, 28, 28, 1)
    img_batch = np.expand_dims(img, 0)

    # Obliczamy wektor latentny i rekonstrukcję
    latent_vec = encoder.predict(img_batch)
    reconstructed = decoder.predict(latent_vec)

    # Wektor latentny wypisujemy na konsolę
    print("Kształt wektora latentnego:", latent_vec.shape)
    print("Wektor latentny (pierwszy element batcha):")
    print(latent_vec[0])

    # Z rekonstrukcji robimy obrazek 2D
    reconstructed_img = reconstructed[0].squeeze()

    # Zapisujemy wynikowy obrazek do pliku
    plt.imsave("reconstructed.png", reconstructed_img, cmap="gray")
    print("Zapisano obrazek wynikowy do pliku 'reconstructed.png'.")

    # Dla wygody pokazujemy też oryginalny i zrekonstruowany obrazek
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title("Oryginalny")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img, cmap="gray")
    plt.title("Po autoenkoderze")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()



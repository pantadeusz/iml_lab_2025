import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


def load_data():
    (x_train, _), (x_test, _) = fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    return x_train, x_test


def rotate_ds(ds):
    # Dodajemy wymiar kanału (28, 28) -> (28, 28, 1)
    # Bez tego RandomRotation bierze ostatni wymiar jako ilość kanałów przez co wynik wychodzi zniekształcony
    ds = tf.expand_dims(ds, -1)

    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(
            factor=(-0.1, 0.1),
            fill_mode='constant',
            fill_value=0.0,  # Wypełniamy czarnym kolorem (0.0)
            interpolation='bilinear'
        )
    ])

    augmented = data_augmentation(ds)

    # Usuwamy wymiar kanału, aby wrócić do formatu (28, 28)
    return tf.squeeze(augmented, axis=-1).numpy()

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvolutionalAutoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(ConvolutionalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape

        # Obliczenie rozmiaru wyjścia po pierwszej warstwie konwolucyjnej (28x28x1 -> 14x14x32)
        # 14 * 14 * 32 = 6272
        self.encoder_flatten_output = 14 * 14 * 32

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),

            # Warstwa Konwolucyjna (zmniejszenie: 28x28 -> 14x14)
            layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'),

            # Spłaszczenie do wektora
            layers.Flatten(),

            # Kompresja do przestrzeni ukrytej (latent_dim)
            layers.Dense(latent_dim, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            # Rozszerzenie wektora do rozmiaru po Flatten w Enkoderze
            layers.Dense(self.encoder_flatten_output, activation='relu'),

            # Przekształcenie wektora z powrotem na kształt 2D (14x14x32)
            layers.Reshape((14, 14, 32)),

            # Konwersja do pełnego obrazka za pomocą Conv2DTranspose
            # Conv2DTranspose - operacja odwrotna (w sensie geometrycznym i kształtu) do standardowej warstwy Conv2D
            layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Usunięcie wymiaru kanału (28, 28, 1) -> (28, 28)
        return tf.squeeze(decoded, axis=-1)

def augmentate_data(x_train , x_test):
    x_train_rotated = tf.expand_dims(rotate_ds(x_train), -1)
    x_train_original = tf.expand_dims(x_train, -1)
    x_test_rotated = tf.expand_dims(rotate_ds(x_test), -1)
    x_test_original = tf.expand_dims(x_test, -1)
    return x_train_rotated, x_train_original, x_test_rotated, x_test_original

def train_autoencoder(x_train, x_test, conv=True):
    shape = x_test.shape[1:]
    latent_dim = 64


    if conv:
        autoencoder = ConvolutionalAutoencoder(latent_dim, shape)
    else:
        autoencoder = Autoencoder(latent_dim, shape)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    # Augmentacja i dodanie kanału
    x_train_rotated, x_train_original, x_test_rotated, x_test_original = augmentate_data(x_train, x_test)

    autoencoder.fit(x_train_rotated, x_train_original,
                    epochs=10,
                    shuffle=True,
                    validation_data=(x_test_rotated, x_test_original))

    encoded_imgs = autoencoder.encoder(x_test_rotated).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    autoencoder.encoder.save("encoder_model.keras")
    autoencoder.decoder.save("decoder_model.keras")

    # show_reconstruction oczekuje (N, H, W), a nie (N, H, W, 1)
    show_reconstruction(tf.squeeze(x_test_rotated, axis=-1), tf.squeeze(decoded_imgs, axis=-1))


def show_reconstruction(output_imgs, result_imgs):
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(output_imgs[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(result_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def create_autoencoder(latent_dim=64):  # Usunąłem x_test z argumentów, bo x_test jest globalnie dostępne po load_data
    # Używamy globalnego x_test do określenia kształtu, ale upewniamy się, że to działa
    # Wyciągamy go z load_data, by był dostępny
    _, x_test_dummy = load_data()
    shape = x_test_dummy.shape[1:]

    autoencoder = ConvolutionalAutoencoder(latent_dim, shape)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    return autoencoder


def get_model(latent_dim=64):
    if tf.io.gfile.exists('encoder_model.keras'):
        print("Loading existing model from disk.")
        encoder = tf.keras.models.load_model('encoder_model.keras', compile=False)
        decoder = tf.keras.models.load_model('decoder_model.keras', compile=False)

        # Tworzymy instancję autoencodera, aby połączyć ładowane warstwy
        autoencoder = create_autoencoder(latent_dim)

        # Wymiana wytrenowanych warstw
        autoencoder.encoder = encoder
        autoencoder.decoder = decoder

        return autoencoder
    return None  # Zwróć None, jeśli pliki nie istnieją


def display_and_save_test_image(input_img, decoded_img, encoded_vector):
    """
    Wyświetla wektor ukryty, pokazuje obrazki i zapisuje wynik.
    """
    print("\n--- Wektor Ukryty (Latent Vector) ---")
    print(encoded_vector)  # Wektor ukryty (latent) wypisany na konsoli

    plt.figure(figsize=(10, 4))

    # Oryginalny (obrócony) obraz wejściowy
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(input_img)
    plt.title("Wejście (Obrócony)")
    plt.gray()
    ax1.axis('off')

    # Zrekonstruowany (wyprostowany) obraz wyjściowy
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(decoded_img)
    plt.title("Wyjście (Wyprostowany)")
    plt.gray()
    ax2.axis('off')

    # Zapisuje obrazek do pliku
    plt.savefig("decoder_example.png")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="Image location", required=True)

    args = parser.parse_args()

    x_train, x_test = load_data()

    # Sprawdzenie, czy model jest już wytrenowany i załadowanie go
    autoencoder = get_model()

    if autoencoder is None:
        print("Model nie znaleziony. Rozpoczynanie treningu...")
        train_autoencoder(x_train, x_test, conv=True)


    print("Model załadowany pomyślnie. Przeprowadzanie testu...")

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    # Skalowanie do 28×28
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0


    # Musimy dodać wymiar kanału i partii (batch) przed podaniem do Enkodera
    test_img_input = tf.expand_dims(tf.expand_dims(img, -1), 0)

    # Przetwarzanie
    encoded_img = autoencoder.encoder(test_img_input).numpy()
    decoded_img_output = autoencoder.decoder(encoded_img).numpy()

    # 4. Wyświetlanie i zapisywanie
    # autoencoder(x) zwraca (N, 28, 28), więc decoded_img_output jest (1, 28, 28)

    display_and_save_test_image(
        input_img=img,  # Wejście do wyświetlenia
        decoded_img=decoded_img_output[0],  # Wyjście do wyświetlenia
        encoded_vector=encoded_img[0]  # Wektor ukryty
    )
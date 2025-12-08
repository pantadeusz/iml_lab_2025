import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import scipy.ndimage

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(f"Kształt danych treningowych: {x_train.shape}")
print(f"Kształt danych testowych: {x_test.shape}")

def rotate_images(images, max_angle=15):
    rotated = np.zeros_like(images)
    angles = np.random.uniform(-max_angle, max_angle, size=len(images))
    for i, (img, angle) in enumerate(zip(images, angles)):
        rotated[i] = scipy.ndimage.rotate(img, angle, reshape=False, mode='constant', cval=0)
    return rotated, angles


print("Obracanie obrazków treningowych...")
x_train_rotated, train_angles = rotate_images(x_train, max_angle=15)
print("Obracanie obrazków testowych...")
x_test_rotated, test_angles = rotate_images(x_test, max_angle=15)

x_train_rotated = x_train_rotated[..., np.newaxis]
x_train_target = x_train[..., np.newaxis]
x_test_rotated = x_test_rotated[..., np.newaxis]
x_test_target = x_test[..., np.newaxis]

print(f"Kształt po dodaniu kanału: {x_train_rotated.shape}")

latent_dim = 64


class ConvAutoencoder(Model):
    def __init__(self, latent_dim):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu', name='latent'),
        ], name='encoder')

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(7 * 7 * 64, activation='relu'),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
        ], name='decoder')

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = ConvAutoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

print("\n=== Rozpoczynam trenowanie ===")
history = autoencoder.fit(
    x_train_rotated, x_train_target,
    epochs=15,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_rotated, x_test_target)
)


print("\n=== Zapisywanie modeli ===")

autoencoder.encoder.save('encoder_model.keras')
print("Zapisano enkoder do: encoder_model.keras")

autoencoder.decoder.save('decoder_model.keras')
print("Zapisano dekoder do: decoder_model.keras")

autoencoder.save_weights('autoencoder_weights.weights.h5')
print("Zapisano wagi autoenkodera do: autoencoder_weights.weights.h5")

n = 10
encoded_imgs = autoencoder.encoder(x_test_rotated[:n]).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
plt.figure(figsize=(20, 6))

for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_target[i].squeeze(), cmap='gray')
    if i == 0:
        plt.ylabel("Oryginał", fontsize=12)
    plt.title(f"#{i}")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_rotated[i].squeeze(), cmap='gray')
    if i == 0:
        plt.ylabel("Obrócony", fontsize=12)
    plt.title(f"{test_angles[i]:.1f}°")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs[i].squeeze(), cmap='gray')
    if i == 0:
        plt.ylabel("Wyprostowany", fontsize=12)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.savefig('conv_autoencoder_rotation_results.png', dpi=150, bbox_inches='tight')
print("Zapisano wyniki do: conv_autoencoder_rotation_results.png")
plt.close()

print("\n=== Zakończono! ===")

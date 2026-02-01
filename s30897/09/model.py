import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, models
from tensorflow.keras.datasets import fashion_mnist

(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype('float32') / 255.
x_test = np.expand_dims(x_test, -1).astype('float32') / 255.

latent_dim = 64
image_shape = x_train.shape[1:]

rotation_layer = layers.RandomRotation(factor=0.5)

class Autoencoder(models.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.augmenter = rotation_layer

        #Encoder
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=image_shape),
            layers.Conv2D(32, 3, activation='relu', padding='same', strides=2),
            layers.Conv2D(64, 3, activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu', name='latent_space')
        ], name='Encoder')

        #Dekoder
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(7 * 7 * 64, activation='relu'),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(32, 3, activation='relu', padding='same', strides=2),
            layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same', strides=2)
        ], name='Decoder')

    def call(self, x):
        x_noisy = self.augmenter(x)
        encoded = self.encoder(x_noisy)
        return self.decoder(encoded)

autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

print("Rozpoczęcie treningu Autoenkodera...")
history = autoencoder.fit(x_train, x_train, epochs=10, shuffle=True, validation_data=(x_test, x_test))
print("Trening zakończony.")

class Autoencoder_without_Conv(models.Model):
  def __init__(self, latent_dim):
    super(Autoencoder_without_Conv, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='sigmoid'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder2 = Autoencoder_without_Conv(latent_dim)

autoencoder2.compile(optimizer='adam', loss=losses.MeanSquaredError())

print("Rozpoczęcie treningu Autoenkodera bez Conv2D...")
history2 = autoencoder2.fit(x_train, x_train, epochs=10, shuffle=True, validation_data=(x_test, x_test))
print("Trening zakończony.")


encoder_path = 'encoder.keras'
decoder_path = 'decoder.keras'

autoencoder.encoder.save(encoder_path)
autoencoder.decoder.save(decoder_path)
print(f"\nModele zapisane: {encoder_path} i {decoder_path}")

x_test_noisy = rotation_layer(x_test).numpy()
decoded_imgs = autoencoder(x_test).numpy()
decoded_imgs2 = autoencoder2(x_test).numpy()

n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    for j, (title, img_data) in enumerate([
        ("Oryginalny (Cel)", x_test[i]),
        ("Obrócony (Wejście)", x_test_noisy[i]),
        ("Zrekonstruowany (Wyjście)", decoded_imgs[i])
    ]):
        ax = plt.subplot(3, n, i + 1 + j * n)
        plt.imshow(img_data.squeeze(), cmap='gray')
        plt.title(title, fontsize=8)
        ax.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 6))
for i in range(n):
    for j, (title, img_data) in enumerate([
        ("Oryginalny (Cel)", x_test[i]),
        ("Obrócony (Wejście)", x_test_noisy[i]),
        ("Zrekonstruowany (Wyjście)", decoded_imgs2[i])
    ]):
        ax = plt.subplot(3, n, i + 1 + j * n)
        plt.imshow(img_data.squeeze(), cmap='gray')
        plt.title(title, fontsize=8)
        ax.axis('off')

plt.tight_layout()
plt.show()
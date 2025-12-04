
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras import layers, losses
from keras.datasets import fashion_mnist
from keras.models import Model

physical_devices = tf.config.list_physical_devices('GPU')
try:
  # Disable first GPU
  tf.config.set_visible_devices(physical_devices[1:], 'GPU')
  logical_devices = tf.config.list_logical_devices('GPU')
  # Logical device was not created for first GPU
  assert len(logical_devices) == len(physical_devices) - 1
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


data_augmentation = layers.RandomRotation(
    factor=0.2, 
    fill_mode='constant', 
    fill_value=0.0
)

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print (x_train.shape)
print (x_test.shape)

x_train_tensor = tf.convert_to_tensor(x_train)
x_test_tensor = tf.convert_to_tensor(x_test)

x_train_rotated = data_augmentation(x_train_tensor, training=True).numpy()
x_test_rotated = data_augmentation(x_test_tensor, training=True).numpy()

class Autoencoder(Model):
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
        layers.Flatten(),
        layers.Dense(latent_dim, activation='relu')
    ], name="encoder")
        
    self.decoder = keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(7 * 7 * 8, activation='relu'),
        layers.Reshape((7, 7, 8)),
        layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ], name="decoder")

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


shape = x_test.shape[1:]
latent_dim = 64
autoencoder = Autoencoder(latent_dim, shape)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train_rotated, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test_rotated, x_test))


encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test_rotated[i].squeeze())
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i].squeeze())
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.savefig("autoencoder_test_unrotate_conv.png")

print("Saving models...")
autoencoder.encoder.save('encoder_model.keras')
autoencoder.decoder.save('decoder_model.keras')
print("Models saved successfully.")

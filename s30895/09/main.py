
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow

(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()



class Autoencoder(tf.keras.models.Model):
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
      tf.keras.layers.Reshape(shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class AutoencoderConv(tf.keras.models.Model):
  def __init__(self, latent_dim, shape):
    super(AutoencoderConv, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
      tf.keras.layers.Reshape(shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class AutoencoderConvTransposition(tf.keras.models.Model):
  def __init__(self):
    super(AutoencoderConvTransposition, self).__init__()
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded



def random_rotate_batch(images, angle_range=(-20, 20)):
    rotated = []
    for img in images:
        angle = np.random.uniform(*angle_range)
        rotated_img = rotate(img, angle, reshape=False, mode='nearest')
        rotated.append(rotated_img)
    return np.array(rotated)


def display_autoencoder_results(x_test, x_test_decoded):
  n = 10
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_decoded[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()

shape = x_test.shape[1:]
latent_dim = 64
autoencoder = Autoencoder(latent_dim,shape)

autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

x_train = x_train.astype('float32') / 255. #train labels
x_test = x_test.astype('float32') / 255. #test labels
x_train_rot = random_rotate_batch(x_train)
x_test_rot = random_rotate_batch(x_test)

autoencoder.fit(x_train_rot, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test_rot, x_test))

encoded_imgs = autoencoder.encoder(x_test_rot).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

autoencoder.encoder.save("models/encoder.keras")
autoencoder.decoder.save("models/decoder.keras")

display_autoencoder_results(x_test_rot, decoded_imgs)
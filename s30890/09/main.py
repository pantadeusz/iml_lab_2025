import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1) 
])

# "obrot -> oryginal"
x_train_rot = data_augmentation(x_train)
x_test_rot = data_augmentation(x_test)

latent_dim = 32

# encoder
encoder_inputs = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, 3, activation="relu", padding="same")(encoder_inputs)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Flatten()(x)
latent = layers.Dense(latent_dim, name="latent_vector")(x)

encoder = keras.Model(encoder_inputs, latent, name="encoder")
encoder.summary()

# decoder
decoder_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(decoder_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
decoder.summary()

# autoencoder
autoencoder_inputs = encoder_inputs
encoded = encoder(autoencoder_inputs)
decoded = decoder(encoded)

autoencoder = keras.Model(autoencoder_inputs, decoded, name="autoencoder")
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

#trening
history = autoencoder.fit(
    x_train_rot, 
    x_train,       
    epochs=20,
    batch_size=128,
    validation_data=(x_test_rot, x_test)
)

encoder.save("encoder_model.h5")
decoder.save("decoder_model.h5")
autoencoder.save("autoencoder_full.h5")

print("Modele zapisane jako encoder_model.h5, decoder_model.h5, autoencoder_full.h5")

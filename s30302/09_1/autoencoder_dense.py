import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1)
])

def preprocess_train(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    image = data_augmentation(image, training=True)
    return image, image

def preprocess_test(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image, image

def load_data(batch_size=64):
    (train_ds, test_ds), _ = tfds.load(
        'FashionMNIST',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )

    train_ds = (train_ds
                .map(preprocess_train)
                .shuffle(1000)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    test_ds = (test_ds
               .map(preprocess_test)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))

    return train_ds, test_ds

def build_sequential_autoencoder(latent_dim):
    encoder = keras.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(8, 3, activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(latent_dim, activation='relu', name='latent_vector')
    ])

    decoder = keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(28*28, activation='sigmoid'),
        layers.Reshape((28,28,1))
    ])

    input_img = keras.Input(shape=(28,28,1))
    latent = encoder(input_img)
    output_img = decoder(latent)
    autoencoder = keras.Model(input_img, output_img)

    return autoencoder, encoder, decoder

def train_autoencoder(autoencoder, train_ds, test_ds, epochs):
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(train_ds, validation_data=test_ds, epochs=epochs)
    return history

if __name__ == "__main__":
    train_ds, test_ds = load_data()

    autoencoder, encoder, decoder = build_sequential_autoencoder(64)
    train_autoencoder(autoencoder, train_ds, test_ds, 10)

    autoencoder.save("autoencoder.keras")
    encoder.save("encoder.keras")
    decoder.save("decoder.keras")
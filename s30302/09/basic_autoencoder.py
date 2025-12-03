import keras
import tensorflow_datasets as tfds
import tensorflow as tf
from matplotlib import pyplot as plt


def preprocess(image, label):
    image = tf.image.resize(image, [128,128])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def load_data(batch_size, shuffle_buffer_size):
    (train_ds, test_ds), info = tfds.load(
        'FashionMNIST',
        split=['train','test'],
        as_supervised=True,
        with_info=True
    )

    train_ds = (train_ds.map(preprocess)
                .shuffle(shuffle_buffer_size)
                .batch(batch_size)
                .map(augment_batch)
                .prefetch(tf.data.AUTOTUNE))

    test_ds = (test_ds.map(preprocess)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))

    num_classes = info.features['label'].num_classes

    return train_ds, test_ds, num_classes

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1)
])
def augment_batch(images, labels):
    return data_augmentation(images), images

def build_encoder(latent_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(128,128,1)),
        keras.layers.Conv2D(16, 3, activation='relu', padding='same', strides=2),
        keras.layers.Conv2D(32, 3, activation='relu', padding='same', strides=2),
        keras.layers.Conv2D(64, 3, activation='relu', padding='same', strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(latent_dim, activation='relu')
    ])
    return model

def build_decoder(latent_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(latent_dim,)),
        keras.layers.Dense(16*16*64, activation='relu'),
        keras.layers.Reshape((16, 16, 64)),
        keras.layers.Conv2DTranspose(64, 3, padding='same', activation='relu', strides=2),
        keras.layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=2),
        keras.layers.Conv2DTranspose(16, 3, padding='same', activation='relu', strides=2),
        keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')
    ])
    return model


def autoencoder(encoder, decoder, input_img):
    latent = encoder(input_img)
    output_img = decoder(latent)
    autoencoder = keras.Model(inputs=input_img, outputs=output_img)
    return autoencoder

def autoencoder_training(autoencoder, train_ds, test_ds, epochs):
    autoencoder.compile(
        optimizer='adam',
        loss='mse'
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = autoencoder.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=[early_stop],
        verbose = 1
    )

    autoencoder.save("autoencoder.h5")
    return autoencoder, history

def preprocess_single_fmnist(image):
    image = tf.image.resize(image, [128, 128])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    image = tf.expand_dims(image, axis=-1)
    return image

if __name__ == "__main__":

    train_ds, test_ds, _ = load_data(32, 1000)

    encoder_model = build_encoder(64)
    decoder_model = build_decoder(64)
    input_img = tf.keras.Input(shape=(128, 128, 1))
    autoenc_model = autoencoder(encoder_model, decoder_model, input_img)

    autoenc_model, history = autoencoder_training(autoenc_model, train_ds, test_ds, 20)

    encoder_model.save("encoder.keras")
    decoder_model.save("decoder.keras")
    for i in range(20):
        i = 1+i
        encoder = keras.models.load_model("encoder.keras")
        decoder = keras.models.load_model("decoder.keras")

        dataset = tfds.load('FashionMNIST', split='test', as_supervised=True)


        for img, label in dataset.take(i):
            input_image = preprocess_single_fmnist(img)
            print("Label:", label.numpy())

        latent_vector = encoder(input_image)
        reconstructed_image = decoder(latent_vector)

        plt.figure(figsize=(6, 3))

        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(tf.squeeze(input_image), cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Reconstructed")
        plt.imshow(tf.squeeze(reconstructed_image), cmap='gray')

        plt.show()

        print("Latent vector shape:", latent_vector.shape)
        print(latent_vector.numpy())



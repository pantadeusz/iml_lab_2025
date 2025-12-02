import os.path

import tensorflow as tf
from keras import models
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def load_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],  # dzieli na zbiór testowy i treningowy
        shuffle_files=True,  # dobra praktyką jest przetasowanie zbioru danych treningowych
        as_supervised=True,  # zwraca krotke zamiast słownik
        with_info=True,  # pokazuje info przy ściąganiu
    )

    return (ds_train, ds_test), ds_info


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label  # Musimy zmienić format


def random_invert_pixel(x):
    # Pobieramy wielkość batcha (np. 128)
    batch_size = tf.shape(x)[0]

    # Losujemy decyzje Tylko dla wymiaru batcha
    # Kształt (batch_size, 1, 1, 1) sprawi, że ta sama decyzja
    # zostanie z broadcastowa na wszystkie piksele danego obrazka.
    random_decisions = tf.random.uniform((batch_size, 1, 1, 1))

    # Aplikujemy warunek
    return tf.where(random_decisions < 0.5, 1.0 - x, x)


def augmentate_ds(ds):
    # Oryginalne obrazy 28 x 28

    data_augmentation = tf.keras.Sequential([
        layers.RandomTranslation(height_factor=0.1, width_factor=0.01),
        layers.RandomRotation(0.03),
        layers.Lambda(random_invert_pixel)  # dodajemy funckję pomocniczą jako warstwe
    ])

    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    return ds


def prepare_data(augmentate_train=False, augmentate_test=False):
    (ds_train, ds_test), ds_info = load_data()

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()  # cache'ujemy dla lepszej wydajności
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)  # deklarujemy batcha po shuffle

    if augmentate_train:
        ds_train = augmentate_ds(ds_train)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()

    if augmentate_test:
        ds_test = augmentate_ds(ds_test)

    return ds_train, ds_test


def build_baseline_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


def build_conv_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), ]
    )

    return model


def show_augmented_imgs(ds):
    # Pobierz jeden batch z potoku treningowego
    image_batch, label_batch = next(iter(ds))

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].numpy().squeeze(), cmap='gray')
        plt.axis("off")
    plt.show()


def train_model(ds_train, ds_test, path, conv=False):
    show_augmented_imgs(ds_train)
    if conv:
        model = build_conv_model()
    else:
        model = build_baseline_model()

    model.fit(ds_train, epochs=20, validation_data=ds_test)
    model.save(path)


def evaluate_model(model_path, ds_test):
    model = models.load_model(model_path)
    loss, acc = model.evaluate(ds_test)
    print(f"Model o scieżce: {model_path} ma Loss: {loss:0.3f}, Accuracy: {acc:0.3f}")
    return loss, acc


def experiment_with_augmentation(augmentate_train, augmentate_test, conv, path):
    ds_train, ds_test = prepare_data(augmentate_train, augmentate_test)
    train_model(ds_train, ds_test, path, conv)
    evaluate_model(path, ds_test)


def main():
    # Stwórz katalog do przechowywania modeli
    os.makedirs("models", exist_ok=True)

    experiment_with_augmentation(False,True, False,"models/baseline_model.keras")
    experiment_with_augmentation(True, True, False,"models/augmentate_train_model.keras")
    experiment_with_augmentation(True, True, True,"models/conv_model.keras")


if __name__ == '__main__':
    main()

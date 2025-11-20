import os.path

import tensorflow as tf
from keras import models
import tensorflow_datasets as tfds
from tensorflow.keras import layers


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


def random_invert_img(x, p=0.5):
  if  tf.random.uniform([]) < p:
    x = (255-x)
  return x

def random_invert(factor=0.5):
  return layers.Lambda(lambda x: random_invert_img(x, factor))

def augmentate_ds(ds):
    # Oryginalne obrazy 28 x 28

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    random_invert = random_invert()

    return ds


def prepare_data(augmentate=False):
    (ds_train, ds_test), ds_info = load_data()

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()  # cache'ujemy dla lepszej wydajności
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)  # deklarujemy batcha po shuffle
    ds_train = ds_train.prefetch(
        tf.data.AUTOTUNE)  # dzięki temu kolejny batch będzie "czekał" w buforze - zwiększa wydajność

    if augmentate:
        ds_test = augmentate_ds(ds)
    else:
        ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(128)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

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


def train_baseline_model(ds_train, ds_test):
    model = build_baseline_model()
    model.fit(ds_train, epochs=10, validation_data=ds_test)
    model.save("models/baseline_model.keras")


def evaluate_model(model_path, ds_test):
    model = models.load_model(model_path)
    loss, acc = model.evaluate(ds_test)
    print(f"Model o scieżce: {model_path} ma Loss: {loss:0.3f}, Accuracy: {acc:0.3f}")
    return loss, acc


def main():
    # Stwórz katalog do przechowywania modeli
    os.makedirs("models", exist_ok=True)

    ds_train, ds_test = prepare_data()
    train_baseline_model(ds_train, ds_test)
    evaluate_model("models/baseline_model.keras", ds_test)


if __name__ == '__main__':
    main()

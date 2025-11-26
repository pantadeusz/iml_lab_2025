import tensorflow as tf
import tensorflow_datasets as tfds
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from sklearn.metrics import classification_report


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


def load_data(batch):
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(batch)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test


def create_baseline_model(input_shape, activation="relu", optimizer="adam"):
    model = tf.keras.models.Sequential(
        [
            layers.Flatten(input_shape=input_shape),
            layers.Dense(128, activation=activation),
            layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


def create_conv_model(input_shape, activation="relu", optimizer="adam"):
    model = tf.keras.models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, (3, 3), activation=activation, padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation=activation, padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation=activation),
            layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


def evaluate_model(model, X_data, y_data=None):
    if isinstance(X_data, tf.data.Dataset):
        y_data = np.concatenate([y.numpy() for x, y in X_data], axis=0)
    elif y_data is None:
        raise ValueError("Błąd danych y")

    y_pred = model.predict(X_data)
    y_pred = np.argmax(y_pred, axis=1)

    cr = classification_report(y_data, y_pred)

    return cr


def save_model(model, name):
    if not (os.path.exists("./models")):
        os.mkdir("./models")

    extension = ".keras"
    counter = 0
    save_model_path = f"models/{name}_{counter}{extension}"

    while os.path.exists(save_model_path):
        counter += 1
        save_model_path = f"models/{name}_{counter}{extension}"

    model.save(save_model_path)
    print(f"Model saved successfuly in {save_model_path}")


def load_model(name):
    path = f"./models/{name}.keras"

    if not os.path.exists(path):
        raise FileNotFoundError("Zła ścieżka pliku z modelem: " + path)

    return tf.keras.models.load_model(path)


def get_data_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomRotation(0.05),
            layers.RandomTranslation(
                height_factor=0.1,
                width_factor=0.1,
                fill_mode="constant",
                fill_value=0.0,
            ),
            layers.Lambda(
                lambda x: tf.where(
                    tf.random.uniform((tf.shape(x)[0], 1, 1, 1)) > 0.9, 1.0 - x, x
                )
            ),
        ]
    )


def save_augmentation_preview(dataset, augmentation_layer, filename="aug_preview.png"):
    for images, labels in dataset.take(1):
        augmented_images = augmentation_layer(images, training=True)

        plt.figure(figsize=(12, 6))

        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.imshow(images[i].numpy().squeeze(), cmap="gray")
            plt.title("Oryginał")
            plt.axis("off")

            plt.subplot(2, 5, i + 6)
            plt.imshow(augmented_images[i].numpy().squeeze(), cmap="gray")
            plt.title("Augmentacja")
            plt.axis("off")

        plt.savefig(filename)
        plt.close()
        break

    print(f"Zapisano obraz do {filename}")


def main():
    if len(sys.argv) == 4:
        _, epochs, batch, optimizer = sys.argv
        epochs = int(epochs)
        batch = int(batch)
    else:
        epochs, batch, optimizer = (
            6,
            128,
            "adam",
        )

    ds_train, ds_test = load_data(batch)

    # ZADANIE 1
    baseline_model = create_baseline_model((28, 28), optimizer=optimizer)

    baseline_model.fit(ds_train, epochs=epochs, validation_data=ds_test)

    print(evaluate_model(baseline_model, ds_test))
    model_name = f"{optimizer}_e{epochs}_bs{batch}"
    save_model(baseline_model, model_name)

    # ZADANIE 2
    aug_check = get_data_augmentation()
    save_augmentation_preview(ds_train, aug_check, "test")

    # ZADANIE 3
    baseline_model = load_model("adam_e6_bs128_0")
    report_bl = evaluate_model(baseline_model, ds_test)
    print(report_bl)

    aug_layers = get_data_augmentation()

    def apply_augmentation_wrapper(images, labels):
        return aug_layers(images, training=True), labels

    ds_train_augmented = ds_train.map(
        apply_augmentation_wrapper, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_test_augmented = ds_test.map(
        apply_augmentation_wrapper, num_parallel_calls=tf.data.AUTOTUNE
    )
    raport_aug = evaluate_model(baseline_model, ds_test_augmented)

    print(raport_aug)

    # ZADANIE 4
    aug_bl_model = create_baseline_model((28, 28), optimizer=optimizer)
    aug_bl_model.fit(ds_train_augmented, epochs=epochs, validation_data=ds_test)

    print("ZWYKŁE DANE")
    print(evaluate_model(aug_bl_model, ds_test))

    print("AUGMENTACJA")
    print(evaluate_model(aug_bl_model, ds_test_augmented))

    model_name = f"aug_{optimizer}_e{epochs}_bs{batch}"
    save_model(aug_bl_model, model_name)

    # ZADANIE 6
    conv_model = create_conv_model((28, 28), optimizer=optimizer)
    conv_model.fit(ds_train_augmented, epochs=epochs, validation_data=ds_test)
    print("ZWYKŁE DANE")
    print(evaluate_model(conv_model, ds_test))

    print("AUGMENTACJA")
    print(evaluate_model(conv_model, ds_test_augmented))


if __name__ == "__main__":
    main()

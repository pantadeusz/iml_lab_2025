import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE


def augment_image(image, label):
    # Negatyw
    if tf.random.uniform([]) > 0.5:
        image = 1.0 - image
    # Rotacja
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    # Przesunięcie
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 20, IMG_SIZE + 20)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    return image, label


def load_data(augment=False):
    (ds_train, ds_val, ds_test), info = tfds.load(
        "beans", split=["train", "validation", "test"], as_supervised=True, with_info=True
    )

    def preprocess(img, label):
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds_train = ds_train.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.map(preprocess, num_parallel_calls=AUTOTUNE)

    if augment:
        ds_train = ds_train.map(augment_image, num_parallel_calls=AUTOTUNE)

    return (ds_train.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE),
            ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE),
            ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE))


def show_augmentation_examples():
    print("Generowanie podglądu augmentacji...")
    ds_train, _, _ = load_data(augment=False)

    # Wyciągamy jedną paczkę (64 zdjęcia) i bierzemy pierwsze zdjęcie
    images, labels = next(iter(ds_train))
    img = images[0]
    label = labels[0]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Oryginał")
    plt.axis("off")

    for i in range(3):
        aug_img, _ = augment_image(img, label)
        plt.subplot(1, 4, i + 2)
        plt.imshow(aug_img)
        plt.title(f"Aug {i + 1}")
        plt.axis("off")

    filename = "augmentation_check.png"
    plt.savefig(filename)
    print(f"Zapisano plik podglądowy: {filename}")
    plt.close()


def create_dense_model():
    model = keras.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_cnn_model():
    model = keras.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    show_augmentation_examples()

    ds_train_clean, ds_val, ds_test_clean = load_data(augment=False)
    ds_train_aug, _, _ = load_data(augment=True)

    # POPRAWKA
    ds_test_augmented = ds_test_clean.unbatch().map(
        lambda x, y: (augment_image(x, y)[0], y),
        num_parallel_calls=AUTOTUNE
    ).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # 1. Baseline
    print("\n--- 1. Baseline (Dense, Czyste dane) ---")
    model_baseline = create_dense_model()
    model_baseline.fit(ds_train_clean, validation_data=ds_val, epochs=EPOCHS, verbose=1)
    _, acc_clean = model_baseline.evaluate(ds_test_clean, verbose=0)
    model_baseline.save("model_baseline.keras")

    # 2. Stress Test
    _, acc_stress = model_baseline.evaluate(ds_test_augmented, verbose=0)

    # 3. Dense + Aug
    print("\n--- 3. Dense + Augmentacja ---")
    model_aug = create_dense_model()
    model_aug.fit(ds_train_aug, validation_data=ds_val, epochs=EPOCHS, verbose=1)
    _, acc_aug_clean = model_aug.evaluate(ds_test_clean, verbose=0)
    _, acc_aug_stress = model_aug.evaluate(ds_test_augmented, verbose=0)
    model_aug.save("model_dense_aug.keras")

    # 4. CNN + Aug
    print("\n--- 4. CNN + Augmentacja ---")
    model_cnn = create_cnn_model()
    model_cnn.fit(ds_train_aug, validation_data=ds_val, epochs=EPOCHS, verbose=1)
    _, acc_cnn_clean = model_cnn.evaluate(ds_test_clean, verbose=0)
    _, acc_cnn_stress = model_cnn.evaluate(ds_test_augmented, verbose=0)
    model_cnn.save("model_cnn.keras")

    print("\n" + "=" * 40)
    print("PODSUMOWANIE WYNIKÓW (ACCURACY)")
    print("=" * 40)
    print(f"Baseline      -> Test Czysty:  {acc_clean:.4f}")
    print(f"Baseline      -> Test Aug:     {acc_stress:.4f}")
    print(f"Dense + Aug   -> Test Czysty:  {acc_aug_clean:.4f}")
    print(f"Dense + Aug   -> Test Aug:     {acc_aug_stress:.4f}")
    print(f"CNN + Aug     -> Test Czysty:  {acc_cnn_clean:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
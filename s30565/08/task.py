import tensorflow as tf

EPOCHS_BASELINE = 3
EPOCHS_AUG = 3
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

BASELINE_MODEL_PATH = "mnist_baseline.keras"
AUG_MODEL_PATH = "mnist_augmented.keras"


def print_training_setup(name: str, epochs: int):
    print("\n==============================")
    print(f"Trening: {name}")
    print("==============================")
    print(f"Epoki      : {epochs}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Optymaliz. : Adam")
    print(f"LR         : {LEARNING_RATE}")
    print("==============================\n")



def load_mnist_datasets():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def make_datasets(x_train, y_train, x_val, y_val, x_test, y_test):
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds

def evaluate_model(model: tf.keras.Model, dataset, name: str):
    print(f"\n--- Ewaluacja modelu: {name} ---")
    loss, acc = model.evaluate(dataset, verbose=0)
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    return loss, acc

data_augmentation_layers = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ],
    name="data_augmentation",
)


@tf.function
def augment_image(image, label):
    image = data_augmentation_layers(image, training=True)

    do_negative = tf.random.uniform(()) > 0.5
    image = tf.cond(do_negative, lambda: 1.0 - image, lambda: image)

    return image, label


def make_augmented_dataset(dataset):
    return (
        dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )



def build_baseline_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="mnist_baseline_dense")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_conv_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="mnist_conv")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_save_model(model, train_ds, val_ds, epochs, name, path):
    print_training_setup(name, epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
    )
    model.save(path)
    print(f"Model '{name}' zapisany do pliku: {path}")
    return history


def main():
    # 1. Dane
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_datasets()
    train_ds, val_ds, test_ds = make_datasets(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    # 2. Baseline – bez augmentacji
    baseline_model = build_baseline_model()
    baseline_history = train_and_save_model(
        baseline_model,
        train_ds,
        val_ds,
        epochs=EPOCHS_BASELINE,
        name="baseline (bez augmentacji)",
        path=BASELINE_MODEL_PATH,
    )

    # Ewaluacja baseline na oryginalnym teście
    baseline_test_loss, baseline_test_acc = evaluate_model(
        baseline_model, test_ds, "baseline / test oryginalny"
    )

    # 3. Augmentacja – przykład: najpierw tylko na teście,
    #    żeby sprawdzić, jak baseline radzi sobie z "pomieszanymi" danymi
    test_ds_aug = make_augmented_dataset(test_ds)
    baseline_aug_loss, baseline_aug_acc = evaluate_model(
        baseline_model, test_ds_aug, "baseline / test z augmentacją"
    )

    # 4. Trening modelu na danych z augmentacją
    train_ds_aug = make_augmented_dataset(train_ds)
    aug_model = build_baseline_model()
    aug_history = train_and_save_model(
        aug_model,
        train_ds_aug,
        val_ds,
        epochs=EPOCHS_AUG,
        name="baseline (trenowany z augmentacją)",
        path=AUG_MODEL_PATH,
    )

    # Ewaluacja modelu trenowanego na danych z augmentacją
    aug_test_loss, aug_test_acc = evaluate_model(
        aug_model, test_ds, "augmented model / test oryginalny"
    )
    aug_test_aug_loss, aug_test_aug_acc = evaluate_model(
        aug_model, test_ds_aug, "augmented model / test z augmentacją"
    )

    # 5. Prosty model konwolucyjny
    conv_model = build_conv_model()
    conv_history = train_and_save_model(
        conv_model,
        train_ds_aug,
        val_ds,
        epochs=EPOCHS_AUG,
        name="model konwolucyjny (z augmentacją)",
        path="mnist_conv_augmented.keras",
    )

    conv_test_loss, conv_test_acc = evaluate_model(
        conv_model, test_ds, "conv model / test oryginalny"
    )
    conv_test_aug_loss, conv_test_aug_acc = evaluate_model(
        conv_model, test_ds_aug, "conv model / test z augmentacją"
    )

    print("\n====================================")
    print("PODSUMOWANIE WYNIKÓW (accuracy)")
    print("====================================")
    print(f"Baseline / test oryginalny        : {baseline_test_acc:.4f}")
    print(f"Baseline / test z augmentacją     : {baseline_aug_acc:.4f}")
    print(f"Aug model / test oryginalny       : {aug_test_acc:.4f}")
    print(f"Aug model / test z augmentacją    : {aug_test_aug_acc:.4f}")
    print(f"Conv model / test oryginalny      : {conv_test_acc:.4f}")
    print(f"Conv model / test z augmentacją   : {conv_test_aug_acc:.4f}")
    print("====================================\n")


if __name__ == "__main__":

    main()



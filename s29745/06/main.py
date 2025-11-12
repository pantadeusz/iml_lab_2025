import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
TUNER_EPOCHS = 5


def load_and_prepare_data():
    ds_train, ds_val, ds_test = tfds.load(
        "beans",
        split=["train", "validation", "test"],
        shuffle_files=True,
        as_supervised=True,
    )

    ds_train = ds_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.cache()
    ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.cache()
    ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test


def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMAGE_SIZE)
    return image, label


def build_model(hp):
    hp_initializer = hp.Choice("initializer", ["glorot_uniform", "he_normal"])
    hp_activation = hp.Choice("activation", ["relu", "elu"])
    hp_dropout_rate = hp.Float("dropout_rate", 0.1, 0.4, step=0.1)
    hp_learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    hp_optimizer_name = hp.Choice("optimizer", ["adam", "rmsprop"])

    init = keras.initializers.get(hp_initializer)

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))

    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))

    model.add(
        keras.layers.Conv2D(
            32, 3, padding="same", kernel_initializer=init, activation=hp_activation
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D())

    model.add(
        keras.layers.Conv2D(
            64, 3, padding="same", kernel_initializer=init, activation=hp_activation
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D())

    model.add(
        keras.layers.Conv2D(
            128, 3, padding="same", kernel_initializer=init, activation=hp_activation
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D())

    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dropout(hp_dropout_rate))

    model.add(
        keras.layers.Dense(128, kernel_initializer=init, activation=hp_activation)
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(hp_dropout_rate * 0.5))

    model.add(keras.layers.Dense(3, activation="softmax"))

    if hp_optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=hp_learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def eval_model(model, ds_test):
    print("\n--- Ocena na zbiorze testowym ---")
    results = model.evaluate(ds_test, verbose=1)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    return results


def main():
    print("Wczytywanie i przygotowywanie danych...")
    ds_train, ds_val, ds_test = load_and_prepare_data()

    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=8,
        executions_per_trial=1,
        directory="keras_tuner_dir",
        project_name="beans_classification",
    )

    print("\nRozpoczynanie wyszukiwania hiperparametrów (Tuning)...")
    tuner.search(ds_train, epochs=TUNER_EPOCHS, validation_data=ds_val)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n--- Najlepsze znalezione hiperparametry ---")
    print(f"Initializer: {best_hps.get('initializer')}")
    print(f"Activation: {best_hps.get('activation')}")
    print(f"Dropout Rate: {best_hps.get('dropout_rate'):.3f}")
    print(f"Learning Rate: {best_hps.get('learning_rate'):.5f}")
    print(f"Optimizer: {best_hps.get('optimizer')}")

    print("\nBudowanie finalnego modelu z najlepszymi HP...")
    model = build_model(best_hps)

    print("\nRozpoczynanie finalnego treningu...")

    best_model_path = "best_beans_model.keras"

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        best_model_path, save_best_only=True, monitor="val_accuracy", mode="max", verbose=1
    )

    history = model.fit(
        ds_train, epochs=EPOCHS, validation_data=ds_val, callbacks=[checkpoint_cb]
    )

    print(f"\nŁadowanie najlepszego modelu z pliku: {best_model_path}")
    final_best_model = keras.models.load_model(best_model_path)

    eval_model(final_best_model, ds_test)

if __name__='__main__':
    main()

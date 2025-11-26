import os
import sys

import keras
import keras_tuner as kt
import numpy as np
import tensorflow_datasets as tfds
from keras_tuner.src.backend.io import tf
from sklearn.metrics import classification_report


def get_dataset():
    train_ds, val_ds, test_ds = tfds.load(
        'beans',
        split=['train', 'validation', 'test'],
        as_supervised=True
    )
    return train_ds, val_ds, test_ds

def preprocess(image, label, img_size):
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def prepare_dataset(dataset, batch_size=32, img_size=(128,128), shuffle=False):
    dataset = dataset.map(lambda x, y: preprocess(x, y, img_size))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(1000)

    return dataset

# def create_model(input_shape, initializer='glorot_uniform', activation='relu', optimizer='adam'):
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=input_shape),
#         keras.layers.Dense(128, activation=activation, kernel_initializer=initializer),
#         keras.layers.Dense(64, activation=activation, kernel_initializer=initializer),
#         keras.layers.Dropout(0.2),
#         keras.layers.Dense(32, activation=activation, kernel_initializer=initializer),
#         keras.layers.Dense(16, activation=activation, kernel_initializer=initializer),
#         keras.layers.Dense(3, activation='softmax')
#     ])
#     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model


def build_model(hp, input_shape):
    model = keras.models.Sequential()

    activations = ["relu", "tanh", "elu"]
    initializers = ["glorot_uniform", "he_normal", "he_uniform"]
    optimizers = ["adam", "rmsprop", "sgd"]

    model.add(keras.layers.Flatten(input_shape=input_shape))

    for i in range(hp.Int("num_layers", 2, 6)):
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"units_{i+1}", min_value=16, max_value=512, step=32),
                activation=hp.Choice("activation", activations),
                kernel_initializer=hp.Choice("initializer", initializers),
            )
        )

    if hp.Boolean("dropout"):
        model.add(keras.layers.Dropout(rate=0.25))

    model.add(keras.layers.Dense(3, activation="softmax"))

    opt_choice = hp.Choice("optimizer", optimizers)
    lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    if opt_choice == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif opt_choice == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=lr)

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def save_model(model, path):
    extension = ".keras"
    counter = 1
    save_model_path = f"{path}{extension}"

    while os.path.exists(save_model_path):
        save_model_path = f"{path}_{counter}{extension}"
        counter += 1

    model.save(save_model_path)

def main():
    epochs, max_trials, executions_per_trial = 30, 2, 5

    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
            max_trials = int(sys.argv[2])
            executions_per_trial = int(sys.argv[3])
        except (IndexError, ValueError):
            print("Niepoprawne argumenty. Używam wartości domyślnych: 30, 2, 5")

    train_ds, val_ds, test_ds = get_dataset()
    train_ds = prepare_dataset(train_ds, shuffle = True)
    val_ds = prepare_dataset(val_ds)
    test_ds = prepare_dataset(test_ds)

    sample_img, _ = next(iter(train_ds))
    input_shape = sample_img.shape[1:]


    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, input_shape=input_shape),
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory="tunning",
        overwrite=True,
        project_name=f"{epochs}_epochs_{max_trials}_trials_{executions_per_trial}_executions_per_trial",
    )


    tuner.search(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()

    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred = np.argmax(best_model.predict(test_ds), axis=1)
    print(classification_report(y_true, y_pred))

    save_model(best_model, f"./models/tuner_model_{epochs}_max_trials_{max_trials}_executions_per_trial_{executions_per_trial}")


if __name__ == "__main__":
    main()


import argparse
import os
from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers


def load_data(img_size: int, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list]:
    (ds_train, ds_val, ds_test), info = tfds.load(
        "beans",
        split=["train", "validation", "test"],
        as_supervised=True,
        with_info=True,
    )
    class_names = info.features["label"].names 

    def preprocess(img, label):
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def make_ds(ds):
        return (ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                  .shuffle(1000)
                  .batch(batch_size))

    return make_ds(ds_train), make_ds(ds_val), make_ds(ds_test), class_names


def create_model(img_size: int, initializer: str = "glorot_uniform", activation: str = "relu", optimizer: str = "adam") -> keras.Model:
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Flatten()(inputs)
    x = layers.Dense(256, activation=activation, kernel_initializer=initializer)(x)
    x = layers.Dense(128, activation=activation, kernel_initializer=initializer)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_once(ds_train, ds_val, img_size: int, hp: dict, epochs: int, out_dir: str) -> Tuple[keras.Model, float]:
    model = create_model(img_size, hp["init"], hp["act"], hp["opt"])
    es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, verbose=0, callbacks=[es])
    best_val = max(history.history.get("val_accuracy", [0.0]))
    return model, float(best_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds_train, ds_val, ds_test, class_names = load_data(args.img_size, args.batch)


    hp = {"act": "relu", "init": "glorot_uniform", "opt": "adam"}
    model, val_acc = train_once(ds_train, ds_val, args.img_size, hp, args.epochs, args.out_dir)
    print(f"[VAL] acc={val_acc:.4f} | act={hp['act']} init={hp['init']} opt={hp['opt']}")


    _, test_acc = model.evaluate(ds_test, verbose=0)
    print(f"[TEST] acc={test_acc:.4f}")

    model_path = os.path.join(args.out_dir, "beans_best.keras")
    labels_path = os.path.join(args.out_dir, "labels.txt")
    model.save(model_path)
    with open(labels_path, "w") as f:
        for c in class_names:
            f.write(c + "\n")
    print(f"Zapisano: {model_path} oraz {labels_path}")


if __name__ == "__main__":
    main()



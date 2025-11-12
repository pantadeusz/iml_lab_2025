import os
import sys

import keras
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

def create_model(input_shape, initializer='glorot_uniform', activation='relu', optimizer='adam'):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(128, activation=activation, initializer=initializer),
        keras.layers.Dense(64, activation=activation),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
    else:
        epochs = 10

    train_ds, val_ds, test_ds = get_dataset()
    train_ds = prepare_dataset(train_ds, shuffle = True)
    val_ds = prepare_dataset(val_ds)
    test_ds = prepare_dataset(test_ds)

    sample_img, _ = next(iter(train_ds))
    input_shape = sample_img.shape[1:]

    model = create_model(input_shape)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")

    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    print(classification_report(y_true, y_pred))

    save_model(model, f"./models/v1_{epochs}")


if __name__ == "__main__":
    main()


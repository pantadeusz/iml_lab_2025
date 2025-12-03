# train_baseline.py
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

def main():
    (ds_train, ds_val), ds_info = tfds.load(
        'mnist',
        split=['train[:90%]', 'train[90%:]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = (ds_train.map(normalize_img)
                .cache()
                .shuffle(10000)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    ds_val = (ds_val.map(normalize_img)
              .batch(BATCH_SIZE)
              .cache()
              .prefetch(tf.data.AUTOTUNE))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_val,
        verbose=1
    )

    model.save('baseline.keras')

if __name__ == "__main__":
    main()
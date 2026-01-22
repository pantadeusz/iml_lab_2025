import tensorflow as tf
import tensorflow_datasets as tfds
import os

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

aug_layers = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
])

def augment_data(images, labels):
    images = aug_layers(images)
    batch_size = tf.shape(images)[0]
    mask = tf.random.uniform((batch_size, 1, 1, 1))
    images = tf.where(mask < 0.2, 1.0 - images, images)
    return images, labels

def main():

    (ds_train, ds_val), ds_info = tfds.load(
        'mnist',
        split=['train[:90%]', 'train[90%:]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = (ds_train
                .map(normalize_img)
                .cache()
                .shuffle(10000)
                .batch(BATCH_SIZE)
                .map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    ds_val = ds_val.map(normalize_img).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

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

    history = model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_val,
        verbose=1
    )

    model.save('augmented_model.keras')

if __name__ == "__main__":
    main()
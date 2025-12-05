import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.metrics import classification_report

def normalize_img(image, label):
  return tf.cast(image, tf.float32) / 255., label

def load_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, ds_info

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


def evaluate_model(model, ds_test):
    y_pred_logits = model.predict(ds_test)
    y_pred = np.argmax(y_pred_logits, axis=1)

    y_true = np.concatenate([y for x, y in ds_test], axis=0)

    print(
        f"""
            --- EVALUATION ---
            {classification_report(y_true, y_pred)}
            """)


def save_model(model, path='tensorflow_MNIST_base.keras'):
    model.save(path)
    print(f'Model saved to {path}')


def main():
    ds_train, ds_test, ds_info = load_data()

    model = build_model()

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

    evaluate_model(model, ds_test)

    save_model(model)

if __name__ == '__main__':
    main()
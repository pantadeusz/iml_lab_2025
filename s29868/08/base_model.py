import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import classification_report


def load_mnist_data(batch_size=64, augment=False, augment_func=None):
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label


    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)


    if augment and augment_func:
        ds_train = ds_train.map(augment_func, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    if augment and augment_func:
        ds_test = ds_test.map(augment_func, num_parallel_calls=tf.data.AUTOTUNE)

    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, ds_info


def create_model(input_shape=(28, 28, 1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def evaluate_model(model, ds_test):

    loss, acc = model.evaluate(ds_test, verbose=0)
    print(f" Loss: {loss:.4f}, Accuracy: {acc:.4f}")


    pred_probs = model.predict(ds_test, verbose=0)
    y_pred = np.argmax(pred_probs, axis=1)


    y_true = np.concatenate([y for x, y in ds_test], axis=0)

    print(classification_report(y_true, y_pred))
    return loss, acc


def save_model(model, path='MNIST_base_model.keras'):
    model.save(path)
    print(f'Model saved to {path}')


def main():

    ds_train, ds_test, ds_info = load_mnist_data(batch_size=64)


    model = create_model()

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )


    evaluate_model(model, ds_test)
    save_model(model)


if __name__ == '__main__':
    main()
import tensorflow as tf
import tensorflow_datasets as tfds
from lab08_evaluation_of_model import augment_image
from lab08_baseline_model import normalize_img, build_model, evaluate_model, save_model

def load_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, ds_info

def main():
    ds_train, ds_test, ds_info = load_data()

    model = build_model()

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

    evaluate_model(model, ds_test)

    save_model(model, path='tensorflow_MNIST_augmentation.keras')

if __name__ == '__main__':
    main()
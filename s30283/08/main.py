import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple
DatasetType = tf.data.Dataset

def get_data() -> Tuple[DatasetType, DatasetType]:
    ds_train, ds_test = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True
    )
    return ds_train, ds_test

def prepare_data(ds_train: DatasetType, ds_test: DatasetType) -> Tuple[DatasetType, DatasetType]:
    AUTOTUNE = tf.data.AUTOTUNE
    ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE) \
                       .cache() \
                       .shuffle(len(list(ds_train))) \
                       .batch(32) \
                       .prefetch(AUTOTUNE)
    ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE) \
                     .batch(32) \
                     .cache() \
                     .prefetch(AUTOTUNE)
    return ds_train, ds_test


def normalize_img(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def create_and_evaluate_base_model():
	model = tf.keras.models.Sequential([
	  tf.keras.layers.Flatten(input_shape=(28, 28)),
	  tf.keras.layers.Dense(128, activation='relu'),
	  tf.keras.layers.Dense(10)
	])
	model.compile(
		optimizer=tf.keras.optimizers.Adam(0.001),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
	)
	model.fit(
		ds_train,
		epochs=6,
		validation_data=ds_test,
	)


if __name__ == '__main__':
    ds_train, ds_test = get_data()
    ds_train, ds_test = prepare_data(ds_train, ds_test)
    print('\033[92mSuccess!\033[00m')
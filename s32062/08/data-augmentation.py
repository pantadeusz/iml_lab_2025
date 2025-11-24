import os
from cProfile import label

import tensorflow as tf
import tensorflow_datasets as tfds
from fontTools.misc.cython import returns
from markdown_it.rules_inline import image
from numpy.core.multiarray import result_type
from tensorflow.keras import layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.2)
])


def augment_img(image, label):
    image = data_augmentation(image)

    return image, label


ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE).map(
    augment_img, num_parallel_calls=tf.data.AUTOTUNE
)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


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

model.save('baseline.keras')

def evaluate_on_new_data(model, data):
    results = model.evaluate(data, verbose=0)
    results_dict = {"loss": results[0], "accuracy": results[1]}

    return results_dict


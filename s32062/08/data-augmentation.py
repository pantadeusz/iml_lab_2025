import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

ds_test_size = ds_info.splits['test'].num_examples
val_size = int(0.5 * ds_test_size)
test_size = ds_test_size - val_size
ds_test_shuffle = ds_test.shuffle(ds_test_size)

ds_val = ds_test_shuffle.take(val_size)
ds_final = ds_test_shuffle.skip(val_size)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


# dlaczego poza funkcją:
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomTranslation(0.2, 0.2),
    tf.keras.layers.RandomInvert(0.2)
])

def augment_img(image, label):
    image = data_augmentation(image)

    return image, label

# def preprocess(ds, normalize=True, random_rotation=False, random_invert=False, random_shift=False):
#     if normalize:
#         ds.


ds_train = (ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
            .map(augment_img, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .shuffle(ds_info.splits['train'].num_examples)
            .batch(128)
            .prefetch(tf.data.AUTOTUNE))

# for images, labels in ds_train.take(1):
#     for i in range(5):
#         plt.imshow(images[i])
#         plt.title(labels[i])
#         plt.show()

ds_val = (ds_val.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
          .map(augment_img, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(128)
          .cache()
          .prefetch(tf.data.AUTOTUNE))

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
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
    validation_data=ds_val,
)

model.save('baseline-train-val-aug.keras')

def evaluate_on_new_data(model, ds_test):
    loss, acc = model.evaluate(ds_test)
    results_dict = {"loss": loss, "accuracy": acc}

    return results_dict

# model = load_model('baseline_train_test_augmented.keras')
evaluate_on_new_data(model, ds_test)
# model.save('baseline-.keras')


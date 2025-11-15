import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image

(train_ds, val_ds, test_ds) = tfds.load(
    'beans',
    split=['train', 'validation', 'test'],
    shuffle_files=True,
    as_supervised=True
)

train_ds_iterator = iter(train_ds)
labels = {}
for i in range(64):
    image, label = next(train_ds_iterator)
    label = int(label.numpy())
    if label not in labels:
        labels[label] = i
        image_numpy = tf.cast(image, tf.uint8).numpy()
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(f'leaf_{label}.png')
print(labels)
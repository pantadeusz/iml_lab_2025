import tensorflow as tf
import tensorflow_datasets as tfds
import os

BATCH_SIZE = 64
MODEL_FILENAME = 'cnn_model.keras'


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

def perform_evaluation(model, dataset, dataset_name):
    print(f"\nEwaluacja dla: {dataset_name}")

    loss, acc = model.evaluate(dataset, verbose=0)

    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    return loss, acc


def main():
    model = tf.keras.models.load_model(MODEL_FILENAME)

    ds_test = tfds.load('mnist', split='test', as_supervised=True, shuffle_files=False)

    ds_clean = (ds_test
                .map(normalize_img)
                .batch(BATCH_SIZE)
                .cache()
                .prefetch(tf.data.AUTOTUNE))


    ds_aug = (ds_test
              .map(normalize_img)
              .batch(BATCH_SIZE)
              .map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))


    perform_evaluation(model, ds_clean, "Zbiór Testowy (Czysty)")
    perform_evaluation(model, ds_aug, "Zbiór Testowy (Z Augmentacją)")

if __name__ == "__main__":
    main()
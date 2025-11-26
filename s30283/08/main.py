import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

DatasetType = tf.data.Dataset

def get_data():
    ds_train, ds_test = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True
    )
    return ds_train, ds_test

def prepare_data(ds_train, ds_test):
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

augmentation_pipeline = tf.keras.Sequential([
    # up to 15 degrees (-0.15 to +0.15 * 2 * pi radians)
    tf.keras.layers.RandomRotation(factor=0.15, fill_mode='constant', interpolation='bilinear'),
    tf.keras.layers.RandomTranslation(
        height_factor=0.1, # up to 10% height shift
        width_factor=0.1,
        fill_mode='constant',
    ),
])

def augment_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    
    image = augmentation_pipeline(image, axis=-1)

    # negative transformation
    if tf.random.uniform(()) > 0.5:
        image = 1.0 - image

    return image, label

def prepare_augmented_data(ds_train_orig, ds_test_orig):
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.AUTOTUNE

    ds_train = ds_train_orig.map(
        augment_img, 
        num_parallel_calls=AUTOTUNE
    )
    ds_train = ds_train.shuffle(
        buffer_size=1024
    )
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.prefetch(AUTOTUNE)

    ds_test = ds_test_orig.map(
        augment_img, 
        num_parallel_calls=AUTOTUNE
    )
    ds_test = ds_test.batch(BATCH_SIZE)
    ds_test = ds_test.prefetch(AUTOTUNE)

    return ds_train, ds_test

def plot_and_save_augmented_images(batch_data, labels):
    data = np.asarray(batch_data)
    _, axes = plt.subplots(3, 3, figsize=(8, 8))
    
    for i in range(9):
        if i >= len(data):
            break

        ax = axes[i // 3, i % 3]
        img_to_plot = data[i].squeeze()

        ax.imshow(img_to_plot, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Label: {labels[i]}", fontsize=10)
            
    plt.tight_layout()
    plt.savefig('augmented_images.png')
    plt.show()

def normalize_img(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Normalizes images: `uint8` -> `float32`. Range 0...1"""
    return tf.cast(image, tf.float32) / 255., label

def create_and_train_baseline_model(ds_train, ds_test, steps_per_epoch=None):
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
      epochs=5,
      validation_data=ds_test,
      steps_per_epoch=steps_per_epoch
    )
    return model

def create_and_train_conv_model(ds_train, ds_test, steps_per_epoch=None):

    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(10)
    ])
    model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.fit(
      ds_train,
      epochs=5,
      validation_data=ds_test,
      steps_per_epoch=steps_per_epoch
    )
    return model

def get_combined_ds_train(datasets):
    AUTOTUNE = tf.data.AUTOTUNE
    ds_train_combined = tf.data.Dataset.sample_from_datasets(
        datasets,
        weights=[0.5, 0.5],
        seed=33
    )
    return ds_train_combined.prefetch(AUTOTUNE)

def evaluate_model(model, ds_test):
    metrics = model.evaluate(ds_test, return_dict=True)
    print('Metrics:\n', metrics)
    return metrics

if __name__ == '__main__':
    ds_train, ds_test = get_data()
    ds_train_pure, ds_test_pure = prepare_data(ds_train, ds_test)
    ds_train_augmented, ds_test_augmented = prepare_augmented_data(ds_train, ds_test)
    # batch_data, labels = next(iter(ds_train_augmented))
    # plot_and_save_augmented_images(batch_data, labels)
    
    # baseline_model = create_and_train_baseline_model(ds_train_pure, ds_test_pure)

    # print('[Base] Evaluation on pure test data, trained on pure train data:')
    # evaluate_model(baseline_model, ds_test_pure)

    # print('[Base] Evaluation on augmented test data, trained on pure train data:')
    # evaluate_model(baseline_model, ds_test_augmented)

    ds_train_combined = get_combined_ds_train([ds_train_pure, ds_train_augmented])
    # baseline_model_aug = create_and_train_baseline_model(ds_train_combined, ds_test_pure, steps_per_epoch=120000//32)

    # print('[Base] Evaluation on pure test data, trained on pure combined with augmented train data:')
    # evaluate_model(baseline_model_aug, ds_test_pure)

    # baseline_model.save('baseline_model.keras')
    # baseline_model_aug.save('baseline_model_aug.keras')

    conv_model = create_and_train_conv_model(ds_train_pure, ds_test_pure)

    print('[Conv] Evaluation on pure test data, trained on pure train data:')
    evaluate_model(conv_model, ds_test_pure)

    print('[Conv] Evaluation on augmented test data, trained on pure train data:')
    evaluate_model(conv_model, ds_test_augmented)

    conv_model_aug = create_and_train_conv_model(ds_train_combined, ds_test_pure, steps_per_epoch=120000//32)

    print('[Conv] Evaluation on pure test data, trained on pure combined with augmented train data:')
    evaluate_model(conv_model_aug, ds_test_pure)

    conv_model.save('conv_model.keras')
    conv_model_aug.save('conv_model_aug.keras')
    
    print('\033[92mSuccess!\033[00m')
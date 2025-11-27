import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
batch_size = 64

(ds_train, ds_test), ds_info = tfds.load(
    'fashion_mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    return tf.cast(image, tf.float32)/255., label

def prepare_data(ds_train, ds_test, ds_info):
    ds_train = ds_train.map( normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map( normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_test

def build_basic_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    print(model.summary())
    return model

def build_cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28,28,1)),

        tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    print(model.summary())
    return model

def get_augmentetions_pipeline():
    augmentations_pipeline = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=(-0.1, 0.1)),
        tf.keras.layers.RandomTranslation(height_factor=(-0.05, 0.05),
                                          width_factor=(-0.05, 0.05)),
        tf.keras.layers.Lambda(lambda x: tf.cond(
            tf.random.uniform([], 0, 1) > 0.5,
            lambda: 1.0 - x,
            lambda: x
        ))
    ])
    return augmentations_pipeline

augmentations_pipeline = get_augmentetions_pipeline()

def augment(image, label):
    return augmentations_pipeline(image), label

def augment_batch(images, labels):
    augmented_images = tf.map_fn(
        lambda img: augment(img, labels)[0],   # augment returns (aug_img, label)
        images,
        fn_output_signature=tf.float32
    )
    return augmented_images, labels


def predict_and_generate_metrics(model, dataset, augment_function=None):
    y_true_list = []
    y_pred_list = []

    labels_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    for x_batch, y_batch in dataset:
        if augment_function is not None:
            x_batch, _ = augment_batch(x_batch, y_batch)
        preds = model.predict(x_batch, verbose=0)
        preds_classes = np.argmax(preds, axis=1)
        y_pred_list.append(preds_classes)
        y_true_list.append(y_batch.numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=labels_names)
    return cm, report

def show_metrics(cm, report):
    labels_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print("Classification Report:\n")
    print(report)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_names, yticklabels=labels_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def show_sample_augmented_image(augment_function, dataset):
    for image_batch, _ in dataset.take(1):
        sample_image = image_batch[0]
        sample_image = tf.expand_dims(sample_image, axis=-1)  # shape: (28,28,1)
        break

    plt.imshow(tf.squeeze(sample_image), cmap='gray')
    plt.title("Normal Image")
    plt.axis('off')
    plt.savefig(f"img")
    plt.show()

    augmented_image, _ = augment_function(sample_image, _)
    plt.imshow(tf.squeeze(augmented_image), cmap='gray')
    plt.title("Augmented Image")
    plt.axis('off')
    plt.savefig("img_aug")
    plt.show()

ds_train_prepared, ds_test = prepare_data(ds_train, ds_test, ds_info)

ds_train_aug = (
    ds_train_prepared
    .map(lambda x, y: augment_batch(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .prefetch(tf.data.AUTOTUNE)
)

show_sample_augmented_image(augment, ds_train_prepared)

model = build_cnn_model()

model.fit(
    ds_train_aug,
    epochs=6,
    validation_data=ds_test
)

cm, report = predict_and_generate_metrics(model,ds_test,augment)
show_metrics(cm, report)

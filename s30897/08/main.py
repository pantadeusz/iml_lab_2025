import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import sys


tf.config.run_functions_eagerly(True)
tf.random.set_seed(100)


BATCH_SIZE = 128
ADAM_LR = 0.001


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)


num_train_examples = ds_info.splits['train'].num_examples
num_test_examples = ds_info.splits['test'].num_examples

print(f"Liczba próbek w zbiorze treningowym: {num_train_examples}")
print(f"Liczba próbek w zbiorze testowym: {num_test_examples}")
print(f"Rozmiar Batcha: {BATCH_SIZE}. Kroków na epokę: {num_train_examples / BATCH_SIZE:.0f}")


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

ds_train = (ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .shuffle(ds_info.splits['train'].num_examples)
            .batch(BATCH_SIZE)
            )
ds_test = (ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(BATCH_SIZE)
           .cache()
           )

images_list = []
labels_list = []
for image_batch, label_batch in ds_test.as_numpy_iterator():
    images_list.append(image_batch)
    labels_list.append(label_batch)
test_images_np = np.concatenate(images_list, axis=0)
test_labels_np = np.concatenate(labels_list, axis=0)


def evaluate_model(model, data, labels):
    print("\n--- Ewaluacja Modelu ---")
    if isinstance(data, tf.data.Dataset):
        loss, accuracy = model.evaluate(data, verbose=1)
    else:
        loss, accuracy = model.evaluate(data, labels, verbose=1, batch_size=BATCH_SIZE)
    print(f"  Strata (Loss): {loss:.4f}")
    print(f"  Dokładność (Accuracy): {accuracy:.4f}")
    return loss, accuracy


# ==============================================================================
# 2. Mechanizm Augmentacji
# ==============================================================================

def gwt_augmentation_pipeline():
    return models.Sequential([
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomRotation(factor=0.2)
    ], name="Data_Augmentation_Pipeline")


data_augmentation = gwt_augmentation_pipeline()


def augment_and_negate_dataset(images_np, labels_np):

    def apply_augmentation(image, label):
        image = data_augmentation(tf.expand_dims(image, 0), training=True)[0]

        if np.random.rand() < 0.3:
            image = 1.0 - image

        return image, tf.cast(label, tf.float32)

    ds_temp = (tf.data.Dataset.from_tensor_slices((images_np, labels_np))
               .map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(BATCH_SIZE))

    images_aug_list = []
    labels_aug_list = []
    for images, labels in ds_temp.as_numpy_iterator():
        images_aug_list.append(images)
        labels_aug_list.append(labels)

    images_aug = np.concatenate(images_aug_list, axis=0)
    labels_aug = np.concatenate(labels_aug_list, axis=0)
    if labels_aug.ndim > 1:
        labels_aug = labels_aug.squeeze()

    return images_aug, labels_aug


# ==============================================================================
# 1. & 3. Model Baseline (Dense)
# ==============================================================================

print("\n--- 1. Trening Modelu BASELINE (Dense) ---")


def build_baseline_model(name="Baseline_Dense_Model"):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ], name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(ADAM_LR),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[SparseCategoricalAccuracy(name='Accuracy')])
    return model


baseline_model = build_baseline_model()
baseline_model.summary()
baseline_model.fit(ds_train, epochs=6, validation_data=ds_test, verbose=1)
baseline_model.save('baseline_mnist_model.keras')

print("\n--- 3. Ocena BASELINE na Augmentowanych Danych Testowych ---")
test_images_augmented_np, test_labels_augmented_np = augment_and_negate_dataset(test_images_np, test_labels_np)

loss_baseline_aug, acc_baseline_aug = evaluate_model(baseline_model, test_images_augmented_np, test_labels_augmented_np)

# ==============================================================================
# 4. Model Augmented Baseline (Dense + Augmentacja)
# ==============================================================================

print("\n--- 4. Trening Modelu Augmented Baseline ---")


def build_augmented_baseline_model(name="Augmented_Dense_Model"):
    model = models.Sequential([
        data_augmentation,
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ], name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(ADAM_LR),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[SparseCategoricalAccuracy(name='Accuracy')])
    return model


aug_baseline_model = build_augmented_baseline_model()
aug_baseline_model.summary()
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
aug_baseline_model.fit(ds_train, epochs=10, validation_data=ds_test, callbacks=[early_stopping], verbose=1)
aug_baseline_model.save('augmented_baseline_mnist_model.keras')

loss_aug_baseline_aug, acc_aug_baseline_aug = evaluate_model(aug_baseline_model, test_images_augmented_np,
                                                             test_labels_augmented_np)

# ==============================================================================
# 5. & 6. Model CNN Augmented (CNN + Augmentacja)
# ==============================================================================

print("\n--- 5. & 6. Trening Modelu CNN Augmented ---")


def build_cnn_augmented_model(name="CNN_Augmented_Model"):
    model = models.Sequential([
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ], name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(ADAM_LR),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[SparseCategoricalAccuracy(name='Accuracy')])
    return model


cnn_aug_model = build_cnn_augmented_model()
cnn_aug_model.summary()
cnn_aug_model.fit(ds_train, epochs=10, validation_data=ds_test, callbacks=[early_stopping], verbose=1)
cnn_aug_model.save('cnn_augmented_mnist_model.keras')

loss_cnn_aug_aug, acc_cnn_aug_aug = evaluate_model(cnn_aug_model, ds_test_augmented)

print("\n" + "=" * 50)
print("             FINALNE PODSUMOWANIE WYNIKÓW")
print("=" * 50)
print(f"| {'Model':<20} | {'Acc. (Aug. Test)':<18} |")
print("|" + "-" * 21 + "|" + "-" * 20 + "|")
print(f"| {'Baseline (Brak Aug.)':<20} | {acc_baseline_aug:<18.4f} |")
print(f"| {'Augmented Dense':<20} | {acc_aug_baseline_aug:<18.4f} |")
print(f"| {'CNN Augmented':<20} | {acc_cnn_aug_aug:<18.4f} |")
print("=" * 50)

sys.exit(0)

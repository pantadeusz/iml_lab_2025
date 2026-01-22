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

def evaluate_model(model, data):
    print("\n--- Ewaluacja Modelu ---")
    if isinstance(data, tf.data.Dataset):
        loss, accuracy = model.evaluate(data, verbose=1)
    else:
        loss, accuracy = model.evaluate(data, verbose=1, batch_size=BATCH_SIZE)
    print(f"  Strata (Loss): {loss:.4f}")
    print(f"  Dokładność (Accuracy): {accuracy:.4f}")
    return loss, accuracy

def get_augmentation_pipeline():
    return models.Sequential([
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomRotation(factor=0.2)
    ], name="data_augmentation")


data_augmentation = get_augmentation_pipeline()



def augment_and_negate(image, label):
    image = data_augmentation(tf.expand_dims(image, 0), training=True)[0]


    if tf.random.uniform(()) < 0.3:
        image = 1.0 - image


    return image, label

ds_test_augmented = (
    ds_test
    .unbatch()
    .map(augment_and_negate, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
)


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

loss_baseline_aug, acc_baseline_aug = evaluate_model(baseline_model, ds_test_augmented)

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


loss_aug_baseline_aug, acc_aug_baseline_aug = evaluate_model(aug_baseline_model, ds_test_augmented)
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

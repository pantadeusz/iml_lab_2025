import tensorflow as tf
import tensorflow_datasets as tfds
from utils import normalize_img, data_augmentation_fn, evaluate_model_metrics
from models import create_baseline_model, create_cnn_model, EPOCHS, BATCH_SIZE
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print("Ładowanie i przygotowanie zbioru danych MNIST...")

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

ds_test_normal = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test_normal = ds_test_normal.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

ds_train_normal = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train_normal = ds_train_normal.cache()
ds_train_normal = ds_train_normal.shuffle(ds_info.splits['train'].num_examples)
ds_train_normal = ds_train_normal.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_train_augmented = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE
).map(
    data_augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE
).cache()
ds_train_augmented = ds_train_augmented.shuffle(ds_info.splits['train'].num_examples)
ds_train_augmented = ds_train_augmented.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_test_augmented = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE
).map(
    data_augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

print("\n\n#####################################################")
print("## 2. Model Bazowy (Baseline) - Trening na Normalnych Danych ##")
print("#####################################################")

baseline_model = create_baseline_model(learning_rate=0.001)

print("Rozpoczęcie treningu modelu bazowego...")
baseline_model.fit(
    ds_train_normal,
    epochs=EPOCHS,
    validation_data=ds_test_normal,
)

evaluate_model_metrics(baseline_model, ds_test_normal, "Baseline (Normal Training) na ds_test_normal")
baseline_model.save('mnist_baseline_model.h5')

print("\n\n#####################################################")
print("## 3. Ewaluacja Baseline na Zaugmentowanym Zbiorze Testowym ##")
print("#####################################################")
evaluate_model_metrics(baseline_model, ds_test_augmented, "Baseline (Normal Training) na ds_test_augmented")



print("\n\n#####################################################")
print("## 4. Model Bazowy (Baseline) - Trening na Zaugmentowanych Danych ##")
print("#####################################################")

model_aug = create_baseline_model(learning_rate=0.001)

print("Rozpoczęcie treningu modelu bazowego na danych z AUGMENTACJĄ...")
model_aug.fit(
    ds_train_augmented,
    epochs=EPOCHS,
    validation_data=ds_test_normal,
)

evaluate_model_metrics(model_aug, ds_test_augmented, "Baseline (Augmented Training) na ds_test_augmented")
model_aug.save('mnist_baseline_augmented_trained_model.h5')


print("\n\n#####################################################")
print("## 5. Model Konwolucyjny (CNN) - Trening na Zaugmentowanych Danych ##")
print("#####################################################")

model_cnn = create_cnn_model(learning_rate=0.001)
model_cnn.summary()

print("Rozpoczęcie treningu modelu CNN na danych z AUGMENTACJĄ...")
model_cnn.fit(
    ds_train_augmented,
    epochs=EPOCHS,
    validation_data=ds_test_normal,
)

evaluate_model_metrics(model_cnn, ds_test_augmented, "CNN (Augmented Training) na ds_test_augmented")
model_cnn.save('mnist_cnn_augmented_trained_model.h5')
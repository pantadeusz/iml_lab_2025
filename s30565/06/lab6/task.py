import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt

def preprocess_image(image, label):
    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'beans',
    split=['train', 'validation', 'test'],
    with_info=True,
    as_supervised=True,
)

NUM_CLASSES = ds_info.features['label'].num_classes

BATCH_SIZE = 16
ds_train = ds_train.map(preprocess_image).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess_image).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess_image).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_model(hp):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=(128, 128, 3)))

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_initializer = hp.Choice('initializer', values=['glorot_uniform', 'he_normal'])

    model.add(tf.keras.layers.Conv2D(
        filters=hp.Int('conv_1_filters', min_value=16, max_value=48, step=16),
        kernel_size=3,
        padding='same',
        activation=hp_activation,
        kernel_initializer=hp_initializer
    ))
    model.add(tf.keras.layers.MaxPooling2D(2))

    model.add(tf.keras.layers.Conv2D(
        filters=hp.Int('conv_2_filters', min_value=32, max_value=64, step=16),
        kernel_size=3,
        padding='same',
        activation=hp_activation,
        kernel_initializer=hp_initializer
    ))
    model.add(tf.keras.layers.MaxPooling2D(2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(
        units=hp.Int('dense_units', min_value=64, max_value=128, step=16),
        activation=hp_activation,
        kernel_initializer=hp_initializer
    ))

    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd'])

    if hp_optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='keras_tuner_dir',
                     project_name='beans_classification')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

print("\n--- Rozpoczynanie optymalizacji hiperparametrów ---")
tuner.search(ds_train, epochs=20, validation_data=ds_val, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
--- Zakończono optymalizację ---
Znaleziono następujące optymalne hiperparametry:
Funkcja aktywacji: {best_hps.get('activation')}
Inicjalizator: {best_hps.get('initializer')}
Liczba filtrów w 1. warstwie Conv2D: {best_hps.get('conv_1_filters')}
Liczba filtrów w 2. warstwie Conv2D: {best_hps.get('conv_2_filters')}
Liczba neuronów w warstwie gęstej: {best_hps.get('dense_units')}
Optymalizator: {best_hps.get('optimizer')}
Tempo uczenia: {best_hps.get('learning_rate')}
""")

model = tuner.hypermodel.build(best_hps)

print("\n--- Rozpoczynanie treningu ostatecznego modelu ---")
history = model.fit(ds_train, epochs=50, validation_data=ds_val, callbacks=[stop_early])

val_acc_per_epoch = history.history.get('val_accuracy', [])
if val_acc_per_epoch:
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print(f"Najlepsza epoka: {best_epoch}")
    print(f"Najwyższa dokładność walidacyjna: {max(val_acc_per_epoch):.4f}")
else:
    print("Brak dostępnych danych o val_accuracy w historii treningu.")

print("\n--- Ewaluacja modelu na zbiorze testowym ---")
loss, accuracy = model.evaluate(ds_test)
print(f"Strata na zbiorze testowym: {loss:.4f}")
print(f"Dokładność na zbiorze testowym: {accuracy:.4f}")

print("\n--- Zapisywanie modelu do pliku 'best_beans_model.keras' ---")
model.save('best_beans_model.keras')
print("Model zapisany.")
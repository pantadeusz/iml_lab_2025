import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras_tuner import RandomSearch

def preprocess(image, label):
    image = tf.image.resize(image, [128,128])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def load_data(batch_size, shuffle_buffer_size):
    (train_ds, test_ds), info = tfds.load(
        'beans',
        split=['train','test'],
        as_supervised=True,
        with_info=True
    )

    train_ds = (train_ds.map(preprocess)
                .shuffle(shuffle_buffer_size)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))
    test_ds = (test_ds.map(preprocess)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))

    num_classes = info.features['label'].num_classes

    return train_ds, test_ds, num_classes

#----------------------------------------------------------------------------------------------------
def build_model(hp, num_classes):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(128, 128, 3)))

    for i in range(hp.Int('conv_layers', 1, 3)):
        model.add(keras.layers.Conv2D(
            filters=hp.Int('filters', 32, 128, step=32),
            kernel_size=(3,3),
            activation='relu',
            padding='same'
        ))
        model.add(keras.layers.MaxPooling2D())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        units=hp.Int('units', 64, 512, step=64),
        activation=hp.Choice('activation', ['relu', 'selu'])
    ))
    model.add(keras.layers.Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1)))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    )

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
#----------------------------------------------------------------------------------------------------

def train_dnn(train_ds, test_ds, num_classes):
    tuner = RandomSearch(
        lambda hp: build_model(hp, num_classes),
        objective='val_accuracy',
        max_trials=15,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='beans_dnn_tuning',
    )

    tuner.search(
        train_ds,
        validation_data=test_ds,
        epochs=15,
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nNajlepsze hiperparametry znalezione przez tuner:")
    print(f"conv_layers: {best_hps.get('conv_layers')}")
    print(f"filters: {best_hps.get('filters')}")
    print(f"units: {best_hps.get('units')}")
    print(f"activation: {best_hps.get('activation')}")
    print(f"learning_rate: {best_hps.get('learning_rate')}")
    print(f"dropout_rate: {best_hps.get('dropout_rate')}\n")

    best_model = build_model(best_hps, num_classes)
#----------------------------------------------------------------------------------------------------
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    history = best_model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=30,
        callbacks=[early_stop],
        verbose=1
    )
#----------------------------------------------------------------------------------------------------

    best_model.save("beans_model_final.h5")
    print("Zapisano najlepszy douczony model do pliku: beans_model_final.h5")

    return best_model, history

def evaluate_model(model, test_ds):
    model_name = type(model).__name__
    print(f"Evaluation for {model_name}:")

    acc = model.evaluate(test_ds)[1]

    return acc

if __name__ == "__main__":
    train_ds, test_ds, num_classes = load_data(32, shuffle_buffer_size=1000)
    model_dnn_beans, history = train_dnn(train_ds, test_ds, num_classes)
    acc = evaluate_model(model_dnn_beans, test_ds)
    print(f"\nKońcowa dokładność modelu: {acc:.4f}")


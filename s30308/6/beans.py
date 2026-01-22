import keras
from keras import layers
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.metrics import classification_report
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch


# 1. Przygotowanie datasetu
def preprocess(image, label):
    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def prepare_dataset():
    ds, info = tfds.load("beans", with_info=True, as_supervised=True)
    train_ds = ds["train"].map(preprocess).batch(32).shuffle(1000)
    val_ds = ds["validation"].map(preprocess).batch(32)
    test_ds = ds["test"].map(preprocess).batch(32)
    return train_ds, val_ds, test_ds


class BeansHyperModel(HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(128, 128, 3)))
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                units=hp.Int('units_1', min_value=64, max_value=512, step=64),
                activation=hp.Choice('activation_1', ['relu', 'tanh'])
            )
        )
        model.add(
            layers.Dense(
                units=hp.Int('units_2', min_value=64, max_value=256, step=32),
                activation=hp.Choice('activation_2', ['relu', 'tanh'])
            )
        )
        model.add(layers.Dense(3, activation='softmax'))

        optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model


# 3. Funkcja do wyświetlania raportu klasyfikacji
def evaluate_model_text(model, dataset, class_names):
    y_true = []
    y_pred = []
    for images, labels in dataset:
        preds = model.predict(images)
        pred_labels = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(pred_labels)
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))


def main():
    train_ds, val_ds, test_ds = prepare_dataset()

    # 4a. Stworzenie tunera
    tuner = RandomSearch(
        BeansHyperModel(),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='beans_classification'
    )

    # Szukanie najlepszych hiperparametrów
    tuner.search(train_ds, validation_data=val_ds, epochs=15)

    # Pobranie najlepszego modelu
    best_model = tuner.get_best_models(num_models=1)[0]

    # Ewaluacja modelu
    class_names = ["angular_leaf_spot", "bean_rust", "healthy"]
    evaluate_model_text(best_model, test_ds, class_names)

    # Zapis modelu
    best_model.save("beans_best_model.keras")


if __name__ == "__main__":
    main()

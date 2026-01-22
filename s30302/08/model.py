import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.src.saving import register_keras_serializable
from tensorflow import keras
from keras_tuner import RandomSearch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

    best_model.save("beans_model_final.h5")
    print("Zapisano najlepszy douczony model do pliku: beans_model_final.h5")

    return best_model, history

def evaluate_model(model, test_ds):
    model_name = type(model).__name__
    print(f"Evaluation for {model_name}:")

    acc = model.evaluate(test_ds)[1]

    return acc

#-----------------------------------------------------Koniec labu 6------Początek labu 8-----------------------------------------------------
#--------------------------------------------Agumantacja--------------------------------------------#
def random_invert_img_batch(x, p=0.5):

    if len(x.shape) == 3:
        x = tf.expand_dims(x, 0)
        added_dim = True
    else:
        added_dim = False

    batch_size = tf.shape(x)[0]
    mask = tf.random.uniform([batch_size], 0, 1) < p
    mask = tf.cast(mask, x.dtype)
    mask = tf.reshape(mask, [batch_size, 1, 1, 1])
    x = x * (1 - mask) + (1 - x) * mask

    if added_dim:
        x = tf.squeeze(x, 0)
    return x

random_invert_layer = tf.keras.layers.Lambda(lambda x: random_invert_img_batch(x, p=0.5))

data_augmentation = tf.keras.Sequential([
    random_invert_layer,
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])
def augment_batch(images, labels):
    return data_augmentation(images), labels


# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomRotation(0.1),
#     tf.keras.layers.RandomTranslation(0.1, 0.1),
#     tf.keras.layers.RandomContrast(0.1),
# ])
#
# def augment_batch(images, labels):
#     return data_augmentation(images), labels


#--------------------------------------------Agumantacja--------------------------------------------#
#--------------------------------------------Evaluation--------------------------------------------#
def evaluate_model_on_new_dataset(model, test_ds, model_name=None):
    y_true = []
    y_pred_classes = []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_pred_classes.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if model_name is None:
        model_name = type(model).__name__
    save_path = f"{model_name}_confusion_matrix.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Zapisano macierz pomyłek do: {save_path}")

    plt.show()

    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes))

    print("Accuracy:")
    print(accuracy_score(y_true, y_pred_classes))
#--------------------------------------------Evaluation--------------------------------------------#
#--------------------------------------------ModelUczonyNaAgumentowychDanychZArchitekturąBazowego--------------------------------------------#
def build_and_train_with_augmentation(num_classes, train_ds, test_ds):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(128, 128, 3)))

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(384, activation='relu'))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00015456105),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=30,
        callbacks=[early_stop],
        verbose=1
    )

    model.save("beans_model_best.h5")
    print("Zapisano model: beans_model_best.h5")

    return model, history
#--------------------------------------------ModelUczonyNaAgumentowychDanychZArchitekturąBazowego--------------------------------------------#
#--------------------------------------------ModelStworzonyPodKrok5Labu--------------------------------------------#
def build_custom_cnn(num_classes, train_ds, test_ds):

    model = keras.Sequential([
        keras.layers.Input(shape=(128,128,3)),

        data_augmentation,

        keras.layers.Conv2D(32, (3, 3), padding='same'),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(32, (3, 3), padding='same'),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(128, (3, 3), padding='same'),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(128, (3, 3), padding='same'),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(),

        keras.layers.Dropout(0.2),

        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=30,
        callbacks=[early_stop],
        verbose=1
    )

    model.save("beans_model_custom.h5")
    print("Zapisano model: beans_model_custom.h5")

    return model, history
#--------------------------------------------ModelStworzonyPodKrok5Labu--------------------------------------------#

if __name__ == "__main__":

    train_ds, test_ds, num_classes = load_data(
        batch_size=32,
        shuffle_buffer_size=1000
    )

    baseline_model = tf.keras.models.load_model("beans_model_final.h5")
    model_aug = tf.keras.models.load_model("beans_model_best.h5")
    custom_model = tf.keras.models.load_model("beans_model_custom.h5")

    print("\n--- BASELINE — TEST ---")
    evaluate_model_on_new_dataset(baseline_model, test_ds, model_name="baseline")

    print("\n--- BASELINE — TEST AUG ---")
    test_ds_aug = test_ds.map(augment_batch)
    evaluate_model_on_new_dataset(baseline_model, test_ds_aug, model_name="baseline_aug")

    print("\n--- MODEL AUGMENTED TRAIN ---")
    evaluate_model_on_new_dataset(model_aug, test_ds, model_name="augmented_model")

    print("\n--- CUSTOM CNN ---")
    evaluate_model_on_new_dataset(custom_model, test_ds, model_name="custom_cnn")

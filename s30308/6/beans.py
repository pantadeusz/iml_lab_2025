import keras
from keras import layers
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

def prepare_dataset():
    # 1. Wczytanie danych z tensorflow
    ds, info = tfds.load("beans", with_info=True, as_supervised=True)

    # 2. Podział na dane
    train_ds = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]

    #print(info)

    # 3. Preproccesing danych
    train_ds = train_ds.map(preprocess).batch(32).shuffle(1000)
    val_ds = val_ds.map(preprocess).batch(32)
    test_ds = test_ds.map(preprocess).batch(32)

    return train_ds, val_ds, test_ds


def preprocess(image, label):
    image = tf.image.resize(image, (128, 128)) # zmieniamy 500x500 na 128x128, aby model się szybciej uczył
    image = tf.cast(image, tf.float32) / 255.0 # Typ image to unit8 a sieci neuronowe obsługują float32 lub float16. Dzielimy wartości w skali 255 (Kolor)
    return image, label


def create_model(initializer='glorot_uniform', activation='relu', optimizer='adam'):
    model = keras.Sequential([
        layers.Input(shape=(128,128,3)),
        layers.Flatten(),
        layers.Dense(256, activation),
        layers.Dense(128, activation),
        layers.Dense(3, keras.activations.softmax)
        
    ], name="beans_model")

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def evaluate_model_text(model, dataset, class_names):
    # Pobranie wszystkich obrazów i etykiet z dataset
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images)
        pred_labels = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(pred_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))


def main():
    train_ds, val_ds, test_ds = prepare_dataset()
    model = create_model()

    model.fit(train_ds, epochs=30, validation_data=val_ds)
    model.evaluate(test_ds)

    class_names = ["angular_leaf_spot", "bean_rust", "healthy"]
    evaluate_model_text(model, test_ds, class_names)



if __name__ == "__main__":
    main()

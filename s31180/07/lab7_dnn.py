from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import numpy as np
import tensorflow as tf
import random
import os

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def prepare_data():
    wine = fetch_ucirepo(id=109)

    X = wine.data.features.values
    y = wine.data.targets.values.ravel() - 1

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_test = scaler.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=0.1, random_state=SEED, stratify=y_tr
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(input_shape=(13,)):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),

        tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_and_evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=0
    )

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    val_accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))



def save_model(model, path='wine_tensorflow.keras'):
    model.save(path)
    model_size = os.path.getsize(path)
    print(f"  Rozmiar: {model_size:,} bajtów ({model_size / 1024:.2f} KB)")


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

    model = build_model()

    train_and_evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)

    save_model(model)


if __name__ == "__main__":
    main()
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import sys
from joblib import dump
import tensorflow.keras as keras
import numpy as np


def load_and_prepare_data():
    # fetch dataset
    wine = fetch_ucirepo(id=109)

    X = wine.data.features
    y = wine.data.targets
    y = y - 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # print(wine.metadata)
    # print(wine.variables)

    return X_train, X_test, y_train, y_test


def save_model(model, save_name, model_type):
    if not (os.path.exists("./models")):
        os.mkdir("./models")

    save_model_path = f"./models/{save_name}"

    if model_type == "rfc":
        dump(model, save_model_path)
    elif model_type == "keras":
        model.save(f"{save_model_path}.keras")
    else:
        print("Wrong model type")
        pass

    print(f"Zapisano do {save_model_path}")


def run_rfc_model(X_train, X_test, y_train, y_test, should_save=False):
    rfc = RandomForestClassifier(random_state=41)
    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    print(classification_report(y_test, y_pred))

    if should_save and sys.argv[1] != "_":
        name = sys.argv[1]
        save_model(rfc, name, "rfc")


def create_model(
    X_train,
    initializer="glorot_uniform",
    activation="relu",
    optimizer="adam",
    regularizer=None,
):
    normalizer = keras.layers.Normalization()
    normalizer.adapt(X_train.values)

    model = keras.Sequential(
        [
            normalizer,
            keras.layers.Dense(
                32,
                activation=activation,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
            ),
            keras.layers.Dense(
                8,
                activation=activation,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
            ),
            keras.layers.Dense(
                8,
                activation=activation,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
            ),
            keras.layers.Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    run_rfc_model(X_train, X_test, y_train, y_test, should_save=len(sys.argv) > 1)

    nn_model = create_model(X_train, regularizer="l2")

    nn_model.fit(
        X_train, y_train, epochs=30, batch_size=16, validation_split=0.1, verbose=0
    )

    nn_model.summary()

    y_pred = np.argmax(nn_model.predict(X_test), axis=1)
    cr = classification_report(y_test, y_pred)
    print(cr)
    # with open('pomoc.txt', 'a') as f:
    #    f.write(cr)
    #    f.close()

    if len(sys.argv) > 2:
        name = sys.argv[2]
        save_model(nn_model, name, "keras")


if __name__ == "__main__":
    main()

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import os

RANDOM_STATE = 50
np.random.seed(RANDOM_STATE)
keras.utils.set_random_seed(RANDOM_STATE)


def prepare_data():
    wine = fetch_ucirepo(id=109)
    X = wine.data.features
    y = np.asarray(wine.data.targets).ravel() - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y - 1
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def create_nn_model(input_size, hidden_units=16, tag="none"):
    layers = [keras.Input(shape=(input_size,))]

    reg = None
    if "l2" in tag:
        reg = keras.regularizers.l2(1e-3)
    elif "l1" in tag:
        reg = keras.regularizers.l1(1e-4)

    layers.append(keras.layers.Dense(hidden_units, kernel_regularizer=reg))

    if "batchnorm" in tag:
        layers.append(keras.layers.BatchNormalization())

    layers.append(keras.layers.Activation("relu"))

    layers.append(keras.layers.Dense(3, activation="softmax"))

    model = keras.Sequential(layers)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    early_stop = EarlyStopping(
        monitor="loss", patience=10, restore_best_weights=True, verbose=0
    )
    model.fit(
        X_train, y_train, epochs=200, batch_size=16, callbacks=[early_stop], verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model {model_name} - Test accuracy: {acc * 100:.2f}%")

    y_pred_classes = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred_classes, digits=4))

    save_model(model, model_name)
    return acc


def save_model(model, name):
    os.makedirs("saved_models", exist_ok=True)
    path = f"saved_models/{name}.keras"
    model.save(path, save_format="keras")
    size = os.path.getsize(path)
    print(f"Model saved to {path}, filesize: {size} bytes\n")


def run_all_configs():
    X_train, X_test, y_train, y_test = prepare_data()
    input_size = X_train.shape[1]

    configs = [
        {"hidden": 16, "tag": "none"},
        {"hidden": 16, "tag": "batchnorm_baseline"},
        {"hidden": 16, "tag": "l1"},
        {"hidden": 16, "tag": "l1_batchnorm"},
        {"hidden": 16, "tag": "l2"},
        {"hidden": 16, "tag": "l2_batchnorm"},
        {"hidden": 8, "tag": "l1_batchnorm"},
        {"hidden": 32, "tag": "l2_batchnorm"},
    ]

    for iter, cfg in enumerate(configs, start=1):
        model_name = f"model_{iter}_{cfg['hidden']}u_{cfg['tag']}"
        model = create_nn_model(input_size, hidden_units=cfg["hidden"], tag=cfg["tag"])
        train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name)


if __name__ == "__main__":
    run_all_configs()

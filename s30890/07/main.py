import os
import numpy as np
import joblib

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def file_size_kb(path):
    return os.path.getsize(path) / 1024.0


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    data = load_wine()
    X = data.data.astype("float32")
    y = data.target.astype("int32")
    num_classes = len(np.unique(y))
    input_dim = X.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Kształt danych:")
    print("X_train:", X_train.shape, "X_val:", X_val.shape)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, y_pred)
    print(f"Dokładność RF na walidacji: {rf_acc:.4f}")

    rf_path = "rf_model.pkl"
    joblib.dump(rf, rf_path)
    print(f"Zapisano model RF do: {rf_path} (rozmiar: {file_size_kb(rf_path):.1f} KB)")

    # 3 warstwy: 2 ukryte + wyjściowa (3 klasy -> 3 neurony z softmax)
    basic_model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    basic_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_basic = basic_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        verbose=0,
    )

    basic_val_loss, basic_val_acc = basic_model.evaluate(X_val, y_val, verbose=0)
    print(f"Dokładność sieci (bez normalizacji) na walidacji: {basic_val_acc:.4f}")

    basic_path = "nn_basic.keras"
    basic_model.save(basic_path)
    print(
        f"Zapisano model nn_basic do: {basic_path} "
        f"(rozmiar: {file_size_kb(basic_path):.1f} KB)"
    )

    print("\nwarstwa normalizująca")

    norm_layer = layers.Normalization(input_shape=(input_dim,))
    
    # adapt dopasowuje średnią i odchylenie standardowe do zbioru treningowego
    norm_layer.adapt(X_train)

    norm_model = keras.Sequential(
        [
            norm_layer,
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    norm_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_norm = norm_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        verbose=0,
    )

    norm_val_loss, norm_val_acc = norm_model.evaluate(X_val, y_val, verbose=0)
    print(f"Dokładność sieci (z normalizacją) na walidacji: {norm_val_acc:.4f}")

    norm_path = "nn_norm.keras"
    norm_model.save(norm_path)
    print(
        f"Zapisano model nn_norm do: {norm_path} "
        f"(rozmiar: {file_size_kb(norm_path):.1f} KB)"
    )

    # 8 różnych konfiguracji sieci (rozmiary, l2, dropout).
    configs = [
        {"name": "A_16", "hidden_units": [16], "l2": 0.0, "dropout": 0.0},
        {"name": "B_16_8", "hidden_units": [16, 8], "l2": 0.0, "dropout": 0.0},
        {"name": "C_8", "hidden_units": [8], "l2": 0.0, "dropout": 0.0},
        {"name": "D_8_l2", "hidden_units": [8], "l2": 1e-4, "dropout": 0.0},
        {"name": "E_8_4_l2", "hidden_units": [8, 4], "l2": 1e-4, "dropout": 0.0},
        {"name": "F_8_dropout", "hidden_units": [8], "l2": 0.0, "dropout": 0.2},
        {"name": "G_4", "hidden_units": [4], "l2": 0.0, "dropout": 0.0},
        {"name": "H_4_l2_dropout", "hidden_units": [4], "l2": 1e-4, "dropout": 0.2},
    ]

    for cfg in configs:
        print(f"\n--- Konfiguracja: {cfg['name']} ---")

        norm_layer_exp = layers.Normalization(input_shape=(input_dim,))
        norm_layer_exp.adapt(X_train)

        if cfg["l2"] > 0.0:
            reg = regularizers.l2(cfg["l2"])
        else:
            reg = None

        model = keras.Sequential()
        model.add(norm_layer_exp)

        # warstwy ukryte
        for units in cfg["hidden_units"]:
            model.add(
                layers.Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=reg,
                )
            )
            if cfg["dropout"] > 0.0:
                model.add(layers.Dropout(cfg["dropout"]))
        model.add(layers.Dense(num_classes, activation="softmax"))

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            verbose=0,
        )

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Dokładność na walidacji: {val_acc:.4f}")

        fname = f"nn_exp_{cfg['name']}.keras"
        model.save(fname)
        size = file_size_kb(fname)
        print(f"Zapisano do: {fname} (rozmiar: {size:.1f} KB)")

        if val_acc == 1.0:
            print("-> udało się uzyskać 100% dokladnosci")
        else:
            print("mniej niz 100% dokladnosci")

if __name__ == "__main__":
    main()

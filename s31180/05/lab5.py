from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
import random
import os

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def prepare_data():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_test = scaler.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=SEED
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_rf_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    val_accuracy = model.score(X_val, y_val)
    y_pred = model.predict(X_test)

    print("\n=== Random Forest ===")
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model

def build_dnn_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=1
    )

    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype("int32")

    print("\n=== DNN (TensorFlow / Keras) ===")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model


def build_dnn_tuner_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test):

    def build_model(hp):
        model = keras.Sequential([
            keras.Input(shape=(X_train.shape[1],)),

            keras.layers.Dense(
                units=hp.Int('units_1', min_value=32, max_value=128, step=16, default=16),
                activation='relu'
            ),

            keras.layers.Dropout(
                hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1, default=0.0)
            ),

            keras.layers.Dense(
                units=hp.Int('units_2', min_value=8, max_value=64, step=8, default=8),
                activation='relu'
            ),

            keras.layers.Dropout(
                hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1, default=0.0)
            ),

            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Float('learning_rate', 0.007, 0.01, step=0.001, default=0.001)
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=20,
        overwrite=True,
        directory='.',
        project_name='keras_tuner_lab5',
        seed=SEED
    )

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=1
    )

    best_model = tuner.get_best_models(num_models=1)[0]

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nNajlepsze hiperparametry znalezione przez tuner:")
    print(f"  units_1: {best_hps.get('units_1')} (było: 16)")
    print(f"  units_2: {best_hps.get('units_2')} (było: 8)")
    print(f"  dropout_1: {best_hps.get('dropout_1'):.2f} (było: 0.0)")
    print(f"  dropout_2: {best_hps.get('dropout_2'):.2f} (było: 0.0)")
    print(f"  learning_rate: {best_hps.get('learning_rate'):.6f} (było: 0.001)")

    y_prob = best_model.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype("int32")

    print("\n=== DNN (TensorFlow / Keras) + Keras Tuner ===")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    best_model.save('best_dnn_tuned.keras')
    print("\nNajlepszy model zapisany: 'best_dnn_tuned.keras'")

    return best_model, tuner


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    rf_model = build_rf_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test)
    dnn_model = build_dnn_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test)
    best_model, tuner = build_dnn_tuner_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    main()
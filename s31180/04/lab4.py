from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def prepare_data():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def build_rf_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nRandom Forest Classification Report")
    print(classification_report(y_test, y_pred))

def build_dnn_and_evaluate(X_train, X_test, y_train, y_test):
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
        epochs=20,
        batch_size=32,
        verbose=0
    )

    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype("int32")

    print("\nDNN (TensorFlow / Keras) Classification Report")
    print(classification_report(y_test, y_pred))



def main():
    X_train, X_test, y_train, y_test = prepare_data()
    build_rf_and_evaluate(X_train, X_test, y_train, y_test)
    build_dnn_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
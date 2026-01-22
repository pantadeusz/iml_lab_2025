from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np


def load_data():
    wine = fetch_ucirepo(id=109)

    X = wine.data.features
    y = wine.data.targets

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    le = LabelEncoder()
    y = le.fit_transform(y.values.ravel())

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model(input_shape=(13,), output_units=3):
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        keras.layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(8, activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(4, activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(output_units, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    print(classification_report(y_test, y_pred))

    matrix = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(matrix).plot()
    plt.show()


def save_model(model, model_path):
    model.save(model_path)



def main():

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    model = create_model()

    model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

    evaluate_model(model, X_test, y_test)
    save_model(model, 'nn_model.keras')

if __name__ == "__main__":
    main()
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import keras_tuner as kt
from matplotlib import pyplot as plt


def load_and_preprocess_data():
    mushroom = fetch_ucirepo(id=73)

    X = mushroom.data.features
    y = mushroom.data.targets

    X = X.apply(LabelEncoder().fit_transform)
    y = LabelEncoder().fit_transform(y.values.ravel())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(f"Zbiór treningowy: {X_train.shape}, Zbiór testowy: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def run_base_model(X_train, y_train, X_test, y_test):
    baseModel = RandomForestClassifier(random_state=42)
    baseModel.fit(X_train, y_train)

    y_pred = baseModel.predict(X_test)

    print("Wyniki bazowego modelu (RandomForest): ")
    print(classification_report(y_test, y_pred))

    matrix = tf.math.confusion_matrix(y_test, y_pred)
    matrix = matrix.numpy()
    ConfusionMatrixDisplay(matrix).plot()
    plt.title("Macierz pomyłek bazowego modelu (RandomForest)\n")

def run_nn_model(X_train, y_train, X_test, y_test):
    model = tf.keras.models.Sequential([
        layers.Dense(2, activation='relu'),
        layers.Dense(2, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10)

    loss,accuracy = model.evaluate(X_test, y_test)
    print(f'Poczatkowy nn - Test loss: {loss:.4f}')
    print(f'Poczatkowy nn - Test accuracy: {accuracy:.4f}')

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("Wyniki początkowego modelu (NN): ")
    print(classification_report(y_test, y_pred))

    matrix = tf.math.confusion_matrix(y_test, y_pred)
    matrix = matrix.numpy()
    ConfusionMatrixDisplay(matrix).plot()
    plt.title("Macierz pomyłek początkowego modelu (NN)\n")


def build_tuner_model(hp):
    model = keras.Sequential()

    model.add(keras.layers.Dense(
        units=hp.Int('units_l1', min_value=2, max_value=16, step=2),
        activation='relu',
    ))

    for i in range (hp.Int('num_of_hidden_layers', 0,1)):

        model.add(keras.layers.Dense(
            units=hp.Int(f'units_hidden{i}', min_value=2, max_value=10, step=2),
            activation='relu',
        ))

    model.add(layers.Dense(1,activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def run_tuner_exp(X_train, y_train, X_test, y_test):

    tuner = kt.RandomSearch(
        build_tuner_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
    )

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(X_train, y_train,
                 epochs=10,
                 validation_split=0.2,
                 callbacks=[stop_early])

    print("Top 5 the best: ")
    tuner.results_summary(num_trials=5)



    best_model = tuner.get_best_models(num_models=1)[0]

    loss,accuracy = best_model.evaluate(X_test, y_test)
    print(f"Tuned NN - Test loss: {loss:.4f}")
    print(f"Tuned NN - Test accuracy tunerowy: {accuracy:.4f}")

    y_pred_prob = best_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("Wyniki tuned NN: ")
    print(classification_report(y_test, y_pred))

    matrix = tf.math.confusion_matrix(y_test, y_pred)
    matrix = matrix.numpy()
    ConfusionMatrixDisplay(matrix).plot()
    plt.title("Macierz pomyłek tunned modelu (NN)\n")


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    run_base_model(X_train, y_train, X_test, y_test)
    run_nn_model(X_train, y_train, X_test, y_test)

    run_tuner_exp(X_train, y_train, X_test, y_test)

    plt.show()
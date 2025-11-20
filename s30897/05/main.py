import numpy as np
from keras.src.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam


def load_data():
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    n_features = X.shape[1]

    median_val = np.median(y)
    y = np.where(y >= median_val, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    print("=" * 30)
    print(f"Liczba danych w zbiorze treningowym: {X_train_scaled.shape}")
    print(f"Liczba danych w zbiorze testowym: {X_test_scaled.shape}")
    print("=" * 30)

    return X_train_scaled, X_test_scaled, y_train, y_test, n_features

def Random_Forest(X_train, X_test, y_train, y_test):
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {"accuracy": accuracy, "confusion_matrix": cm, "classification_report": report}

def DNN(X_train, X_test, y_train, y_test, n_features):
    DNN_Model = Sequential()
    DNN_Model.add(Input(shape=(n_features,)))
    DNN_Model.add(Dense(units=64, activation="relu"))
    DNN_Model.add(Dropout(0.3))
    DNN_Model.add(Dense(units=32, activation="relu"))
    DNN_Model.add(Dropout(0.3))
    DNN_Model.add(Dense(units=16, activation="relu"))
    DNN_Model.add(Dense(units=1, activation="sigmoid"))

    DNN_Model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    DNN_Model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    y_pred_prob = DNN_Model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {"accuracy": accuracy, "confusion_matrix": cm, "classification_report": report}

def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(10,)))

    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(
            Dense(
                units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                activation="relu"
            )
        )
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(units=1, activation="sigmoid"))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])


    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model

def DNN_tuner(X_train, X_test, y_train, y_test):
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=60,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='diabetes_classification',
        overwrite=True
    )
    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(
        X_train, y_train,
        epochs=100,
        validation_split=0.2,
        callbacks=[stop_early],
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("---------Best Hyperparameters:---------")
    print(f"Liczba warstw: {best_hps.get('num_layers')}")
    for i in range(best_hps.get('num_layers')):
        print(f"Warstwa {i+1}: {best_hps.get(f'units_{i}')} neuronów, dropout: {best_hps.get(f'dropout_{i}')}")
    print(f'Learning rate: {best_hps.get('learning_rate')}')

    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2 ,verbose=1)

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {"accuracy": accuracy, "confusion_matrix": cm, "classification_report": report, "best_hps": best_hps}

def compare_models(rf_metrics, dnn_metrics, dnn_tuned_metrics):
    print("=" * 30)
    print(f"\nRANDOM FOREST\nAccuracy: {rf_metrics['accuracy']:.3f}\n=========Macierz Pomyłek:=========\n{rf_metrics['classification_report']}\n=========Raport Klasyfikacji:=========\n{rf_metrics['confusion_matrix']}\n")
    print("=" * 30)
    print(f"\nSIEĆ NEURONOWA (DNN)\nAccuracy: {dnn_metrics['accuracy']:.3f}\n=========Macierz Pomyłek:=========\n{dnn_metrics['classification_report']}\n=========Raport Klasyfikacji:=========\n{dnn_metrics['confusion_matrix']}\n")
    print("=" * 30)
    print(f"\nSIEĆ NEURONOWA Z KERAS TUNER:\nAccuracy: {dnn_tuned_metrics['accuracy']:.3f}\n=========Macierz Pomyłek:=========\n{dnn_tuned_metrics['confusion_matrix']}\n=========Raport Klasyfikacji:=========\n{dnn_tuned_metrics['classification_report']}\n")

    results = {
        'Random Forest': rf_metrics['accuracy'],
        'DNN Bazowy': dnn_metrics['accuracy'],
        'DNN z Tunerem': dnn_tuned_metrics['accuracy'],
    }

    print("\nRanking modeli:")
    for i, (model, accuracy) in enumerate(sorted(results.items(), key=lambda item: item[1], reverse=True), 1):
        print(f"{i}. {model}: {accuracy:.3f}")
    print("=" * 30)
if __name__ == '__main__':
    X_train, X_test, y_train, y_test, n_features = load_data()

    print("\n[1/3] Trenowanie Random Forest...")
    rf_metrics = Random_Forest(X_train, X_test, y_train, y_test)
    print("\n[2/3] Trenowanie DNN Bazowego...")
    dnn_metrics = DNN(X_train, X_test, y_train, y_test, n_features)
    print("\n[3/3] Trenowanie DNN z Tunerem...")
    dnn_tuned_metrics = DNN_tuner(X_train, X_test, y_train, y_test)
    compare_models(rf_metrics, dnn_metrics, dnn_tuned_metrics)
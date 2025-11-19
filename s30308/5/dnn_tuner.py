import kagglehub
import pandas as pd
from tensorflow.keras import layers, models, optimizers, losses
import keras
from keras_tuner import RandomSearch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# 1. Wczytywanie i dzielenie danych
def split_data_for_training(test_size=0.2, random_state=42):
    path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
    data = pd.read_csv(path + "/diabetes.csv")

    X = data.drop('Outcome', axis=1).values
    y = data['Outcome'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# 2. Budowa modelu DNN
def build_dnn_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(8,)))  # liczba cech w danych

    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=8, max_value=128, step=8),
            activation=hp.Choice('activation', ['relu', 'tanh'])
        ))

    model.add(layers.Dense(1, activation='sigmoid'))

    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model



# 3. Trenowanie modelu DNN
def train_dnn_model(model, X_train, y_train, epochs=300, batch_size=10):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return history


# 4. Ewaluacja DNN
def evaluate_dnn(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"[DNN] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)

    plot_confusion_matrix(y_test, y_pred_classes, title="DNN - Macierz pomyłek")

    return accuracy


# 5. Model Random Forest
def train_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(random_state=7, n_estimators=100)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[Random Forest] Accuracy: {acc:.4f}")
    plot_confusion_matrix(y_test, y_pred, title="Random Forest - Macierz pomyłek")

    return acc


# 6. Funkcja pomocnicza do rysowania macierzy pomyłek
def plot_confusion_matrix(y_true, y_pred, title="Macierz pomyłek"):
    # Macierz pomyłek
    cm = confusion_matrix(y_true, y_pred)

    # Raport klasyfikacji
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Rysowanie w matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title(title)

    axes[1].axis('off')  # wyłącz oś
    axes[1].table(cellText=report_df.round(2).values,
                  colLabels=report_df.columns,
                  rowLabels=report_df.index,
                  loc='center')
    axes[1].set_title("Raport klasyfikacji")

    plt.tight_layout()

    plt.show()

def find_best_model():
    tuner = RandomSearch(
        build_dnn_model,
        objective='val_accuracy',
        max_trials=50,
        executions_per_trial=5,
        directory='my_tuner_dir',
        project_name='dnn_tuning'
    )

    tuner.search(
        X_train, y_train,
        epochs=20,
        validation_split=0.2
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Najlepsze hiperparametry:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")

    return best_model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_data_for_training()

    # Trening i testowanie DNN
    dnn_model = find_best_model() # Szukamy najlepszych parametrów dla modelu
    train_dnn_model(dnn_model, X_train, y_train)
    acc_dnn = evaluate_dnn(dnn_model, X_test, y_test)

    # Trening i testowanie Random Forest
    acc_rf = train_random_forest(X_train, X_test, y_train, y_test)

    # Porównanie wyników
    print("\n=== Porównanie modeli ===")
    print(f"DNN Accuracy: {acc_dnn:.4f}")
    print(f"Random Forest Accuracy: {acc_rf:.4f}")

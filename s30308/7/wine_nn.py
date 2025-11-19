from wine_classic import prepare_dataset
import keras
from keras import layers
import os
import shutil

# Lista eksperymentów: (liczba_neuronów, dropout, L2)
configs = [
    (8, 0.3, 0.0005),
    (8, 0.25, 0.001),
    (8, 0.2, 0.001),
    (8, 0.3, 0.001), # oryginalny najlepszy
    (8, 0.3, 0.0001),
    (6, 0.3, 0.001),
    (4, 0.3, 0.001),
    (6, 0.25, 0.0005)
]



def build(neurons, dropout, l2, normalizer):
    model = keras.Sequential()
    model.add(layers.Input(shape=(13,)))
    model.add(normalizer)
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(neurons, activation="relu", kernel_regularizer=keras.regularizers.l2(l2)))
    model.add(layers.Dense(3, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model_with_configs(repeats=1):
    X_train, X_test, y_train, y_test = prepare_dataset()

    normalizer = keras.layers.Normalization()
    normalizer.adapt(X_train.to_numpy())

    os.makedirs("models", exist_ok=True)

    outcomes = []
    for repeat in range(0, repeats):
        for i, (neurons, dropout, l2) in enumerate(configs, start=0):
            model = build(neurons, dropout, l2, normalizer)  # nowy model za każdym razem
            model.fit(X_train, y_train, epochs=30, verbose=True)

            loss, acc = model.evaluate(X_test, y_test)

            print(f"Run {i + 1}: Loss = {loss:.3f}, Accuracy = {acc:.2f}")
            outcomes.append({"Run": i + 1, "Loss": loss, "Accuracy": acc})

            if acc == 1.0:  # jeżeli accuracy jest równe 100
                model.save(f"models/model_run{i + 1}.keras")

        print(outcomes)



def save_best_model():
    best_model_path = "best_model.keras"
    models_dir = "models"

    if not os.listdir(models_dir):
        print("Nie można zapisać modelu, bo żaden nie osiągnął 100% dokładności.")
        return

    # Rozmiar najlepszego modelu
    try:
        best_size = os.path.getsize(best_model_path)
    except FileNotFoundError:
        best_size = float('inf')

    print(f"Rozmiar najlepszego modelu ({best_model_path}): {best_size / 1024:.2f} KB")

    model_files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]
    sizes = []

    for f in model_files:
        path = os.path.join(models_dir, f) # ścieżka do modelu
        size = os.path.getsize(path) # rozmiar modelu
        sizes.append((f, size)) # dodajemy krotke do listy
        print(f"Model {f} ma rozmiar {size / 1024:.2f} KB")

    # Znalezienie najmniejszego modelu
    if sizes:
        smallest_model = min(sizes, key=lambda x: x[1])
        print(f"Najmniejszy model w folderze {models_dir}: {smallest_model[0]}, {smallest_model[1] / 1024:.2f} KB")

        if smallest_model[1] < best_size:
            src_path = os.path.join(models_dir, smallest_model[0])
            shutil.copy(src_path, best_model_path)
            print(f"Najmniejszy model został zapisany jako {best_model_path}")
        else:
            print(f"Najmniejszym modelem nadal pozostaje {best_model_path}")



def main():
    train_model_with_configs(5)
    save_best_model()


if __name__ == "__main__":
    main()

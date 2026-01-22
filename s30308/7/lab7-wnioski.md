# Wnioski po eksperymentach z modelem FNN

## Pierwszy eksperyment z siecią neuronową (Przykład 3)

Mój model:

````
model = keras.Sequential()
    model.add(layers.Input(shape=(13,)))
    model.add(layers.Dense(26, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
````

Wynik: Loss = 0.512, Accuracy = 0.78 

(Wyniki się chwieją od 40% do 80% dokładności).

## Drugi eksperyment: dodanie warstwy normalizującej

Model po dodaniu normalizaci:
```
def build(normalizer):
    model = keras.Sequential()
    model.add(layers.Input(shape=(13,)))
    model.add(normalizer)
    model.add(layers.Dense(26, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
```

Przed budową modelu musimy przypasować dane do normalizera.

Po dodaniu tej warstwy - dokładność modelu jest w granicach od 95% do 100%

**Wielkość pliku modelu: 30 KB (prawie 7 razy mniejszy niż klasyfikator Random Forest)**

## Trzeci eksperement: dodanie regularyzacji

Najlepszy model o 100% dokładności udało się osiągnąć za pomocą:
- Liczba neuronów: 8
- Dropout: 0.3
- L2-rate: 0.001

**Ostateczny rozmiar modelu: 27.67 KB**
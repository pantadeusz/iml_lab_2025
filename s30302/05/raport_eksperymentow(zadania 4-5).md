# Zadania 4 i 5 – Eksperymenty z Keras Tunerem i analiza wyników

## 1. Opis architektury modelu DNN

Model był budowany dynamicznie przy użyciu funkcji `build_model(hp)` oraz autotunera Keras Tuner (`RandomSearch`).

W procesie strojenia tuner dobierał następujące hiperparametry:
- **Liczba warstw ukrytych** (`num_layers`): od 1 do 3  
- **Liczba neuronów w warstwie** (`units`): od 32 do 512, krok 32  
- **Funkcja aktywacji** (`activation`): `relu` lub `tanh`  
- **Współczynnik uczenia** (`learning_rate`): od `1e-4` do `1e-2`, próbkowany logarytmicznie  
- **Optymalizator** (`optimizer`): `adam`, `rmsprop` lub `sgd`

Warstwa wyjściowa była stała:  
`Dense(3, activation='softmax')`, dopasowana do klasyfikacji 3 klas w zbiorze Iris.

Model był trenowany z funkcją straty:
`sparse_categorical_crossentropy`  
oraz metryką oceny:
`accuracy`.

---

## 2. Opis eksperymentu

Celem eksperymentu było dostrojenie modelu DNN przy użyciu **autotunera Keras Tuner**, tak aby uzyskać wynik lepszy lub porównywalny do modelu bazowego (RandomForestClassifier z Scikit-Learn).

W każdym eksperymencie zmieniano liczbę prób (`max_trials`) — czyli liczbę kombinacji hiperparametrów testowanych przez tuner.  
Dla każdego zestawu wykonano pełne trenowanie i ocenę modelu na zbiorze testowym.

Eksperymenty zostały przeprowadzone dla trzech wartości:
- `max_trials = 5`
- `max_trials = 10`
- `max_trials = 15`

Każdy eksperyment zakończono zapisaniem najlepszego modelu (`best_dnn_model.keras`).

---

## 3. Parametry autotunera

Użyty tuner: `keras_tuner.RandomSearch`

Parametry konfiguracyjne:
```python
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=<5|10|15>,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='iris_dnn_tuning',
)
```

---

## 4. Wyniki eksperymentów

###  Dla `max_trials = 5`

**Najlepsze hiperparametry:**
- num_layers: 1  
- units: 96  
- activation: tanh  
- learning_rate: 0.0010316  

**Wynik:**
- Baseline (RandomForest): `accuracy = 1.0`
- DNN (tuned): `accuracy = 0.97`

---

###  Dla `max_trials = 10`

**Najlepsze hiperparametry:**
- num_layers: 2  
- units: 288  
- activation: tanh  
- learning_rate: 0.0014032  

**Wynik:**
- Baseline (RandomForest): `accuracy = 1.0`
- DNN (tuned): `accuracy = 1.0`

---

###  Dla `max_trials = 15`

**Najlepsze hiperparametry:**
- num_layers: 3  
- units: 320  
- activation: relu  
- learning_rate: 0.0049507  

**Wynik:**
- Baseline (RandomForest): `accuracy = 1.0`
- DNN (tuned): `accuracy = 1.0`

---

## 5. Wnioski

- Model **DNN z autotunerem Keras Tuner** uzyskał bardzo wysokie wyniki dokładności — od 0.97 do 1.0.  
- Wraz ze wzrostem liczby prób (`max_trials`) tuner był w stanie dobrać lepsze hiperparametry (więcej warstw, więcej neuronów), co przełożyło się na poprawę wyniku do poziomu identycznego jak model RandomForest.
- Ostatecznie **zarówno DNN, jak i RandomForest osiągnęły 100% dokładności**, co oznacza, że zbiór Iris jest na tyle prosty, że obie metody są w stanie go perfekcyjnie sklasyfikować.
- Choć nie udało się „przebić” modelu Scikit, **eksperyment zakończył się sukcesem**, bo proces strojenia hiperparametrów został przeprowadzony poprawnie i skutecznie.

---

## 6. Podsumowanie

| Parametr eksperymentu | max_trials=5 | max_trials=10 | max_trials=15 |
|------------------------|---------------|----------------|----------------|
| Liczba warstw          | 1             | 2              | 3              |
| Neurony w warstwie     | 96            | 288            | 320            |
| Aktywacja              | tanh          | tanh           | relu           |
| Learning rate          | 0.00103       | 0.00140        | 0.00495        |
| Dokładność DNN         | 0.97          | 1.00           | 1.00           |
| Dokładność Baseline    | 1.00          | 1.00           | 1.00           |

---

 **Wniosek końcowy:**  
Autotuning DNN przy pomocy Keras Tunera pozwolił na pełną optymalizację modelu do poziomu modelu RandomForest. Proces był skuteczny, stabilny i zgodny z założeniami zadania.  

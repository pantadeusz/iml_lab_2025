Przestrzeń poszukiwań hiperparametrów obejmowała:
*   `units_1` (neurony w pierwszej warstwie): od 16 do 64 (krok co 8).
*   `dropout` (współczynnik dropout): od 0.1 do 0.5 (krok co 0.1).
*   `units_2` (neurony w drugiej warstwie): od 8 do 32 (krok co 8).
*   `learning_rate` (współczynnik uczenia): `1e-3` lub `1e-4`.

**Model bazowy: `RandomForestClassifier`**
```
              precision    recall  f1-score   support

           0       0.98      0.93      0.95        43
           1       0.96      0.99      0.97        71

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114
```

**Model zoptymalizowany: `Deep Neural Network`**
```
              precision    recall  f1-score   support

           0       1.00      0.98      0.99        43
           1       0.99      1.00      0.99        71

    accuracy                           0.99       114
   macro avg       0.99      0.99      0.99       114
weighted avg       0.99      0.99      0.99       114
```

tuner ewidentnie znalazł lepsze parametry niz klasyczny rf, wyniki mówia same za siebie


wnioski z LLMa, czy w taki sposób ma wyglądać podsumowanie eksperymentu?
-------------
# Lab 5: Optymalizacja modelu DNN za pomocą Keras Tuner

W ramach laboratorium przeprowadzono eksperyment polegający na porównaniu skuteczności klasycznego
modelu uczenia maszynowego (`RandomForestClassifier`) z siecią neuronową (DNN), której hiperparametry zostały zoptymalizowane
przy użyciu narzędzia Keras Tuner. Celem było zbadanie, czy zoptymalizowana sieć DNN osiągnie lepsze wyniki na zbiorze danych
"Breast Cancer" z biblioteki Scikit-learn.

### Ogólny opis architektury

Zbudowano model głębokiej sieci neuronowej (DNN) składający się z następujących warstw:
1.  **Warstwa wejściowa**: Przyjmująca dane o kształcie `(30,)`, co odpowiada liczbie cech w zbiorze danych.
2.  **Pierwsza warstwa gęsta (Dense)**: Z funkcją aktywacji `ReLU`. Liczba neuronów w tej warstwie była hiperparametrem do strojenia.
3.  **Warstwa Dropout**: Zapobiegająca przeuczeniu. Współczynnik "porzucania" neuronów był również hiperparametrem.
4.  **Druga warstwa gęsta (Dense)**: Z funkcją aktywacji `ReLU`. Liczba neuronów była hiperparametrem.
5.  **Warstwa wyjściowa (Dense)**: Z jednym neuronem i funkcją aktywacji `sigmoid`, odpowiednią dla problemu klasyfikacji binarnej.

Model był kompilowany z użyciem optymalizatora `Adam`, a jako funkcję straty wybrano `binary_crossentropy`.

### Ogólny opis eksperymentu

Eksperyment przeprowadzono na zbiorze danych dotyczącym raka piersi. Dane zostały podzielone na zbiór treningowy (80%) i testowy (20%).
Cechy zostały przeskalowane za pomocą `StandardScaler`.

1.  **Model bazowy (baseline)**: Wytrenowano klasyfikator `RandomForestClassifier` na przeskalowanych danych treningowych.
2.  **Model DNN**: Użyto `Keras Tuner` do znalezienia najlepszej kombinacji hiperparametrów dla modelu sieci neuronowej.
Po zakończeniu strojenia, najlepszy znaleziony model został użyty do predykcji na zbiorze testowym.
3.  **Porównanie**: Wyniki obu modeli na zbiorze testowym zostały porównane przy użyciu raportu klasyfikacji `classification_report`.

### Parametry autotunera

Do strojenia modelu wykorzystano `RandomSearch` z biblioteki `keras-tuner`.
Skonfigurowano go do przetestowania **10 różnych kombinacji** hiperparametrów (`max_trials=10`).
Każda próba była trenowana przez **20 epok**.

Przestrzeń poszukiwań hiperparametrów obejmowała:
*   `units_1` (neurony w pierwszej warstwie): od 16 do 64 (krok co 8).
*   `dropout` (współczynnik dropout): od 0.1 do 0.5 (krok co 0.1).
*   `units_2` (neurony w drugiej warstwie): od 8 do 32 (krok co 8).
*   `learning_rate` (współczynnik uczenia): `1e-3` lub `1e-4`.

### Wyniki

Poniżej przedstawiono raporty klasyfikacji dla obu modeli na zbiorze testowym.

**Model bazowy: `RandomForestClassifier`**
```
              precision    recall  f1-score   support

           0       0.98      0.93      0.95        43
           1       0.96      0.99      0.97        71

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114
```

**Model zoptymalizowany: `Deep Neural Network`**
```
              precision    recall  f1-score   support

           0       1.00      0.98      0.99        43
           1       0.99      1.00      0.99        71

    accuracy                           0.99       114
   macro avg       0.99      0.99      0.99       114
weighted avg       0.99      0.99      0.99       114
```

### Wnioski

Eksperyment zakończył się sukcesem. Zoptymalizowany model oparty o głęboką sieć neuronową (DNN)
uzyskał lepsze wyniki niż model bazowy `RandomForestClassifier` we wszystkich kluczowych metrykach.
Dokładność (`accuracy`) wzrosła z **96,5%** do **99,1%**. Co więcej, model DNN osiągnął niemal idealne wyniki
precyzji i czułości dla obu klas, co świadczy o jego wysokiej zdolności do generalizacji. Użycie Keras Tuner
pozwoliło na systematyczne i zautomatyzowane znalezienie wydajnej architektury sieci, co pokazuje,
że nawet przy ograniczonym czasie na eksperymenty, autotuning jest potężnym narzędziem do poprawy jakości modeli głębokiego uczenia.
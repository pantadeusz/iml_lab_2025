# Eksperyment z autotunerem Keras Tuner i porównanie z Random Forest (Scikit-learn)

## 1. Opis architektury modelu DNN

Model zbudowano w oparciu o klasyczną sieć neuronową typu **Dense Neural
Network (DNN)** do klasyfikacji binarnej.\
Architektura została ustalana dynamicznie przez **Keras Tuner**, który
decydował o liczbie warstw i liczbie neuronów.

-   **Wejście**: 8 cech (z danych Pima Indians Diabetes)
-   **Warstwy ukryte**: od 1 do 3 warstw
-   **Liczba neuronów w warstwie**: od 8 do 128 (krok 8)
-   **Funkcje aktywacji**: `relu` lub `tanh`
-   **Warstwa wyjściowa**: 1 neuron z aktywacją `sigmoid`
-   **Optymalizator**: Adam
-   **Funkcja straty**: binary crossentropy

------------------------------------------------------------------------

## 2. Opis eksperymentu

Celem eksperymentu było porównanie jakości klasyfikacji pomiędzy
klasycznym modelem **Random Forest** a zoptymalizowaną siecią neuronową
**DNN** dostrojoną przy pomocy **Keras Tuner (RandomSearch)**.

Etapy eksperymentu: 1. Wczytano dane z zestawu
`Pima Indians Diabetes Dataset` (Kaggle). 2. Dane zostały podzielone na
zbiory treningowy (80%) i testowy (20%). 3. Zastosowano standaryzację
(`StandardScaler`). 4. Użyto **RandomSearch** z parametrami: -
`max_trials = 50` - `executions_per_trial = 5` - `epochs = 20`,
`validation_split = 0.2` 5. Najlepszy model został następnie douczony
(300 epok, batch_size=10). 6. Porównano końcowe wyniki z klasycznym
modelem **Random Forest** (`n_estimators=100`).

------------------------------------------------------------------------

## 3. Parametry autotunera

  Parametr                 Zakres / wartość
  ------------------------ -------------------
  `num_layers`             1 -- 3
  `units_i`                8 -- 128 (krok 8)
  `activation`             relu / tanh
  `learning_rate`          1e-2, 1e-3, 1e-4
  `objective`              val_accuracy
  `max_trials`             50
  `executions_per_trial`   5

------------------------------------------------------------------------

### Najlepsze hiperparametry (z autotunera)

    Best val_accuracy: 0.7854
    Total elapsed time: 00h 20m 41s

    num_layers: 2
    units_0: 112
    activation: tanh
    learning_rate: 0.01
    units_1: 96
    units_2: 48

------------------------------------------------------------------------

## 4. Wyniki eksperymentu

### Wynik zoptymalizowanego modelu DNN

    Loss: 1.9350
    Accuracy: 0.6948

### Wynik modelu bazowego (Random Forest)

    Accuracy: 0.7208

### Porównanie

  Model           Accuracy
  --------------- ----------
  DNN (tuned)     0.6948
  Random Forest   0.7208

------------------------------------------------------------------------

## 5. Wnioski

Eksperyment z użyciem **Keras Tuner** został przeprowadzony poprawnie, a
tuner działał prawidłowo

Najlepszy zestaw hiperparametrów wskazał architekturę z dwiema warstwami
ukrytymi (`112` i `96` neuronów, `tanh`, `lr=0.01`).\
Pomimo tego, finalna sieć osiągnęła **gorszy wynik (0.6948)** niż model
bazowy **Random Forest (0.7208)**.

Wynik ten sugeruje, że: - dane mają niewielką liczbę cech i ograniczoną
złożoność, przez co Random Forest skuteczniej modeluje relacje
nieliniowe, - DNN może wymagać większej liczby danych lub regularizacji
(dropout, early stopping), - w przypadku prostych zbiorów tablicowych
metody oparte o drzewa decyzyjne często przewyższają modele neuronowe.

# Raport z Eksperymentu: Optymalizacja DNN vs. Random Forest

**Cel eksperymentu:** Dostrojenie modelu głębokiej sieci neuronowej (DNN) przy użyciu Keras Tuner, aby osiągnąć lepszą dokładność klasyfikacji niż bazowy model Random Forest na zbiorze danych diabetes.

---

## Ogólny Opis Architektury

Przestrzeń poszukiwań dla modelu DNN obejmowała następujące parametry:

**Struktura warstw:**
- Warstwa wejściowa: 10 cech (dane diabetes)
- Liczba warstw ukrytych: 1-4 warstwy Dense
- Liczba neuronów w każdej warstwie: od 32 do 512 (z krokiem 32)
- Funkcja aktywacji: ReLU dla wszystkich warstw ukrytych
- Regularyzacja: Dropout po każdej warstwie ukrytej (współczynnik 0.1-0.5, krok 0.1)
- Warstwa wyjściowa: 1 neuron z aktywacją sigmoid (klasyfikacja binarna)

**Optymalizacja:**
- Optymalizator: Adam
- Learning rate: wybierany z wartości [1e-3, 1e-4, 1e-5]
- Funkcja straty: binary_crossentropy
- Metryka: accuracy

---

## Ogólny Opis Eksperymentu

**Model bazowy (baseline):**
Wytrenowano RandomForestClassifier z 100 drzewami (random_state=42) jako punkt odniesienia.

**Proces eksperymentu:**

1.**Modyfikacja parametrów tunera:** Oryginalna konfiguracja (max_trials=20, executions_per_trial=2) testowała tylko 20 unikalnych kombinacji hiperparametrów. W celu rzetelnego przeszukania przestrzeni parametry zostały zmienione na:
   - `max_trials = 60` 
   - `executions_per_trial = 1` (więcej unikalnych architektur)

2.**Kontrola czasu:** Przed uruchomieniem finalnego eksperymentu oszacowano czas pojedynczego uczenia. Dzięki mechanizmowi EarlyStopping (patience=5), każda próba trwała średnio 5-10 sekund, co dało szacunkowy całkowity czas ~5-10 minut dla 60 prób.

3.**Cel:** Znalezienie konfiguracji DNN przewyższającej dokładność Random Forest.

---

## Parametry Autotunera

- **Strategia przeszukiwania:** `keras_tuner.RandomSearch`
- **Cel optymalizacji:** `val_accuracy` (maksymalizacja dokładności na zbiorze walidacyjnym)
- **Liczba prób:** 60 (`max_trials=60`)
- **Liczba wykonań na próbę:** 1 (`executions_per_trial=1`)
- **Liczba epok:** 100
- **Walidacja:** 20% zbioru treningowego jako validation_split
- **Early stopping:** patience=5, monitorowanie `val_loss`
- **Katalog wyników:** `tuner_results/diabetes_classification`
- **Tryb:** overwrite=True (czyszczenie poprzednich wyników)

---

## Wyniki

### Podsumowanie Dokładności

| Model | Accuracy |
|-------|----------|
| Random Forest (Baseline) | 0.742 |
| DNN (Keras Tuner) | 0.764 |
| **Różnica** | **+0.022 (+2.2 pp)** |

### Model Bazowy: Random Forest

**Accuracy:** 0.742

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.73      0.71      0.72        42
           1       0.76      0.77      0.76        47

    accuracy                           0.74        89
   macro avg       0.74      0.74      0.74        89
weighted avg       0.74      0.74      0.74        89
```

### Zoptymalizowany Model: DNN (Keras Tuner)

**Accuracy:** 0.764

**Najlepsze znalezione hiperparametry:**
- Liczba warstw ukrytych: 2
- Warstwa 1: 320 neuronów, dropout: 0.2
- Warstwa 2: 32 neurony, dropout: 0.1
- Learning rate: 0.001

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.75      0.74      0.74        42
           1       0.78      0.79      0.78        47

    accuracy                           0.76        89
   macro avg       0.76      0.76      0.76        89
weighted avg       0.76      0.76      0.76        89
```

---

# Wnioski

Eksperyment zakończył się sukcesem! 
-

Zoptymalizowany model DNN osiągnął dokładność 0.764, przewyższając Random Forest (0.742) o 2.2 punktu procentowego. Kluczowym czynnikiem było zwiększenie liczby prób tunera z 20 do 60, co umożliwiło rzetelne przeszukanie przestrzeni hiperparametrów. Wyniki pokazują, że nawet na małych, tabelarycznych zbiorach danych, gdzie dominują modele drzewiaste, odpowiednio dostrojona sieć neuronowa może być konkurencyjna. Wcześniejsze próby z mniejszą liczbą trials dawały wyniki gorsze lub porównywalne z baseline, co potwierdza konieczność systematycznego przeszukiwania przestrzeni hiperparametrów.


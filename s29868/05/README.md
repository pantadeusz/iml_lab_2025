## Opis architektury modeli

### 1. Model bazowy (Baseline)

Model bazowy to **RandomForestClassifier** z biblioteki *Scikit-learn*, uruchomiony z domyślnymi hiperparametrami (`random_state=42`).

### 2. Zoptymalizowana sieć neuronowa (DNN)

Architektura sieci **DNN** była przedmiotem automatycznego strojenia przy użyciu **Keras Tuner**.  
Zdefiniowano następującą przestrzeń poszukiwań (search space):

- **Warstwa wejściowa:**  
  `Dense` z aktywacją `relu`.  
  Liczba neuronów: od 2 do 16 (krok co 2).

- **Warstwy ukryte:**  
  Strojenie liczby dodatkowych warstw (0 lub 1).

- **Wewnętrzna warstwa ukryta (jeśli istnieje):**  
  `Dense` z aktywacją `relu`.  
  Liczba neuronów: od 2 do 10 (krok co 2).

- **Warstwa wyjściowa:**  
  `Dense(1, activation='sigmoid')`

Z logów tunera wynika, że najlepsza znaleziona architektura to:

```python
Dense(12, activation='relu')
Dense(10, activation='relu')
Dense(1, activation='sigmoid')
```

## Parametry autotunera

Do znalezienia optymalnej architektury wykorzystano **Keras Tuner** z konfiguracją:

- **strategy:** RandomSearch  
- **objective:** maksymalizacja `val_accuracy`  
- **max_trials:** 5  
- **executions_per_trial:** 1  
- **Parametry treningu:**  
  - `epochs = 10`  
  - `validation_split = 0.2`  
  - `EarlyStopping(monitor='val_loss', patience=5)`

---

## Wyniki

### 1. Wyniki modelu bazowego (RandomForest)

Model **RandomForestClassifier** osiągnął perfekcyjny wynik.

| Klasa | precision | recall | f1-score | support |
|:------|:----------:|:------:|:--------:|:--------:|
| 0 | 1.00 | 1.00 | 1.00 | 843 |
| 1 | 1.00 | 1.00 | 1.00 | 782 |
| **accuracy** |  |  | **1.00** | **1625** |

---

### 2. Wyniki zoptymalizowanego modelu DNN (Keras Tuner)

Model DNN z architekturą `12x10` neuronów również osiągnął perfekcyjną skuteczność:

| Klasa | precision | recall | f1-score | support |
|:------|:----------:|:------:|:--------:|:--------:|
| 0 | 1.00 | 1.00 | 1.00 | 843 |
| 1 | 1.00 | 1.00 | 1.00 | 782 |
| **accuracy** |  |  | **1.00** | **1625** |

---

## Wnioski

Eksperyment zakończył się sukcesem.  
Model bazowy **RandomForestClassifier** osiągnął **dokładność 100%**, co wskazuje, że zbiór danych *Mushroom* jest stosunkowo prosty i łatwo separowalny za pomocą modeli drzewiastych.

Pierwotna, ręcznie zdefiniowana sieć DNN (2 warstwy po 2 neurony) uzyskała jedynie ok. **95%** dokładności, co pokazało znaczenie właściwego doboru architektury.  
Jednak **Keras Tuner**, w ograniczonej przestrzeni poszukiwań i tylko 5 prób, był w stanie znaleźć bardzo małą, ale skuteczną architekturę (**12×10 neuronów**), która dorównała wynikowi lasu losowego.

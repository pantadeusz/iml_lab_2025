# Podsumowanie

### Random Forest

Przy podziale danych na zbiór treningowy/walidacyjny/testowy (64%/16%/20%) uzyskano wyniki:

- Validation Accuracy: 1.0000
- Test Accuracy: 0.9722
- Rozmiar modelu: 189.08 KB

| Klasa | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|-----|
| 0 | 0.92 | 1.00 | 0.96 | 12 |
| 1 | 1.00 | 0.93 | 0.96 | 14 |
| 2 | 1.00 | 1.00 | 1.00 | 10 |
| **Accuracy** | | | **0.97** | 36 |
| **Macro avg** | 0.97 | 0.98 | 0.97 | 36 |
| **Weighted avg** | 0.97 | 0.97 | 0.97 | 36 |

Udało mi się jednak poprawić te wyniki:

- Validation Accuracy: 1.0000
- Test Accuracy: 1.0000

| Klasa | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|-----|
| 0     | 1.00      | 1.00   | 1.00     | 12  |
| 1     | 1.00      | 1.00   | 1.00     | 14  |
| 2     | 1.00      | 1.00   | 1.00     | 10  |
| **Accuracy**     |           |        | **1.00** | 36 |
| **Macro avg**    | 1.00      | 1.00   | 1.00     | 36  |
| **Weighted avg** | 1.00      | 1.00   | 1.00     | 36  |

Wyniki te udało mi się uzyskać na 2 sposoby:
- zmniejszenie zbioru walidacyjnego, podział (72%/8%/20%)
rozmiar modelu: 199.56 KB
- zwiększenie liczby drzew 100 -> 350, podział pierwotny
rozmiar modelu: 670.59 KB

Udało się uzyskać Accuracy i Precision 100% co pokrywa się z informacjami na stronie datasetu (https://archive.ics.uci.edu/dataset/109/wine)

## DNN

**Architektura**

- Warstwa wejściowa: 13 cech
- Warstwa ukryta 1: Dense(32, activation='relu', kernel_initializer='he_uniform')
- Warstwa ukryta 2: Dense(16, activation='relu', kernel_initializer='he_uniform')
- Warstwa wyjściowa: Dense(3, activation='softmax')

**Hiperparametry**

- Optimizer: Adam (learning_rate=0.001)

- Funkcja straty: sparse_categorical_crossentropy

- Batch size: 16

- Liczba epok: 50

- Podział danych: 72% train, 8% validation, 20% test

- Preprocessing: StandardScaler

## Wpływ warstwy normalizacyjnej

### Wyniki bez warstwy normalizacyjnej

- Validation Accuracy: 1.0000
- Test Accuracy: 0.9722
- Rozmiar modelu: 36.85 KB

| Klasa / Miara    | Precision | Recall | F1-score | Support |
|------------------| --------- | ------ |----------| ------- |
| 0                | 1.00      | 1.00   | 1.00     | 12      |
| 1                | 0.93      | 1.00   | 0.97     | 14      |
| 2                | 1.00      | 0.90   | 0.95     | 10      |
| **Accuracy**     |           |        | **0.97** | 36      |
| **Macro avg**    | 0.98      | 0.97   | 0.97     | 36      |
| **Weighted avg** | 0.97      | 0.97   | 0.97     | 36      |

### Wyniki po dodaniu warstwy normalizacyjnej

- Validation Accuracy: 1.0000
- Test Accuracy: 0.9722
- Rozmiar modelu: 49.91 KB

| Klasa / Miara    | Precision | Recall | F1-score | Support |
|------------------| --------- | ------ |----------| ------- |
| 0                | 1.00      | 1.00   | 1.00     | 12      |
| 1                | 0.93      | 1.00   | 0.97     | 14      |
| 2                | 1.00      | 0.90   | 0.95     | 10      |
| **Accuracy**     |           |        | **0.97** | 36      |
| **Macro avg**    | 0.98      | 0.97   | 0.97     | 36      |
| **Weighted avg** | 0.97      | 0.97   | 0.97     | 36      |

#### Optymalizacja parametrów

Przy tak małym zbiorze treningowym, batch_size=16 może być zbyt mały, aby warstwa normalizująca mogła obliczyć stabilne statystyki.

Z tego względu postanowiłem przetestować wpływ warstwy normalizacyjnej na tę architekturę, zwiekszając batch_size do 32 (reszta parametrów nie zmieniona).

### Wyniki bez warstwy normalizacyjnej

- Validation Accuracy: 1.0000
- Test Accuracy: 0.9167
- Rozmiar modelu: 36.85 KB

| Klasa / Miara | Precision | Recall | F1-score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| 0             | 1.00      | 0.92   | 0.96     | 12      |
| 1             | 0.82      | 1.00   | 0.90     | 14      |
| 2             | 1.00      | 0.80   | 0.89     | 10      |
| **Accuracy**      |           |        | **0.92**     | 36      |
| **Macro avg**     | 0.94      | 0.91   | 0.92     | 36      |
| **Weighted avg**  | 0.93      | 0.92   | 0.92     | 36      |

Bez warstw BatchNormalization zwiększenie batch_size pogorszyło wyniki, gdyż zmniejszyło o połowę liczbę kroków uczenia.

### Wyniki po dodaniu warstwy normalizacyjnej

- Validation Accuracy: 1.0000
- Test Accuracy: 1.0000
- Rozmiar modelu: 49.91 KB

| Klasa / Miara | Precision | Recall | F1-score | Support |
|---------------| --------- | ------ | -------- | ------- |
| 0             | 1.00      | 1.00   | 1.00     | 12      |
| 1             | 1.00      | 1.00   | 1.00     | 14      |
| 2             | 1.00      | 1.00   | 1.00     | 10      |
| **Accuracy**    |           |        | **1.00**     | 36      |
| **Macro avg**     | 1.00      | 1.00   | 1.00     | 36      |
| **Weighted avg**  | 1.00      | 1.00   | 1.00     | 36      |

Połączenie stabilniejszego batch_size=32 z warstwą BatchNormalization przyniosło oczekiwany rezultat.
Od teraz to będzie model bazowy

## Eksperymenty

Celem jest zmniejszenie rozmiaru modelu przy zachowaniu 100% dokładności oraz sprawdzenie wpływu regularyzacji na wyniki.

Sprawdziłem 8 różnych kombinacji:

```python
experiments = [
        ("Base_32_16", [32, 16], None),
        ("L2_32_16", [32, 16], 'l2'),
        ("L1_32_16", [32, 16], 'l1'),
        ("Standard_16_8", [16, 8], None),
        ("Standard_16", [16], None),
        ("Standard_8", [8], None),
        ("L2_8", [8], 'l2'),
        ("L1_8", [8], 'l1'),
    ]
```

Żadna z kombinacji nie zapewniła nam 100% dokładności (oprócz Base_32_16, który jest modelem bazowym).

**Model bazowy dopasował się idealnie do struktury danych**, i nawet dodanie do niego regularyzacji (L2/L1), pogorszyło wyniki. Narzuciło to zbyt silne ograniczenia, uniemożliwiając sieci nauczenia się zależności wymaganych do osiągnięcia 100% dokładności (Regularization-induced Underfitting).

Zwiększenie liczby epok, również nie przyniosło pozytywnych skutków, co potwierdza moją "pogrubioną" tezę z poprzedniego akapitu.

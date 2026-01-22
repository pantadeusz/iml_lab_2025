### RandomForest
Ranfom Forest zgodnie z oczekiwaniami osiągnął 100% accuracy

### Neural Network

#### Wyniki przed dodaniem warstwy normalizacyjnej
- warstwa wejściowa
- dense 1: 32 neurony, relu
- dense 2: 16 neuronów, relu
- dense 3 (wyjściowa): 3 neurony, softmax
- epoki: 50
- batch size: 16


| Class          | Precision | Recall | F1-score | Support |
|----------------|-----------|--------|----------|---------|
| 0              | 1.00      | 1.00   | 1.00     | 12      |
| 1              | 0.93      | 1.00   | 0.97     | 14      |
| 2              | 1.00      | 0.90   | 0.95     | 10      |
| **Accuracy**   |           |        | 0.97     | 36      |
| **Macro avg**  | 0.98      | 0.97   | 0.97     | 36      |
| **Weighted avg** | 0.97    | 0.97   | 0.97     | 36      |

#### Eksperymentując z epokami postanowiłem dodać early stop, aby uniknąć dobierania ilości epok to każdej sieci

Wynik poprzedniego modelu po dodaniu early stop:

- Early stop zadziałał na 126 epoce, ale w tym przypadku wyniki mamy takie same

| Class           | Precision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| 0               | 1.00      | 1.00   | 1.00     | 12      |
| 1               | 0.93      | 1.00   | 0.97     | 14      |
| 2               | 1.00      | 0.90   | 0.95     | 10      |
| **Accuracy**    |           |        | 0.97     | 36      |
| **Macro avg**   | 0.98      | 0.97   | 0.97     | 36      |
| **Weighted avg**| 0.97      | 0.97   | 0.97     | 36      |

#### Dodanie warstwy normalizacyjnej i eksperymenty z rozmiarem sieci

- Struktura modelu taka sama jak poprzednio, ale między warstwami ukrytymi dodałem **BatchNormalization()**


| Class           | Precision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| 0               | 1.00      | 1.00   | 1.00     | 12      |
| 1               | 1.00      | 1.00   | 1.00     | 14      |
| 2               | 1.00      | 1.00   | 1.00     | 10      |
| **Accuracy**    |           |        | 1.00     | 36      |
| **Macro avg**   | 1.00      | 1.00   | 1.00     | 36      |
| **Weighted avg**| 1.00      | 1.00   | 1.00     | 36      |


- Jak widać, model osiągnąl idealne wyniki

#### Teraz zmniejszam rozmiar modelu

**Najmniejszy model jaki osiągał 100% accuracy w moim przypadku miał jedną warstwę ukrytą z 16 neuronami i BatchNormalization, od teraz będzie on punktem odniesienia**

| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| 0                | 1.00      | 1.00   | 1.00     | 12      |
| 1                | 1.00      | 1.00   | 1.00     | 14      |
| 2                | 1.00      | 1.00   | 1.00     | 10      |
| **Accuracy**     |           |        | 1.00     | 36      |
| **Macro avg**    | 1.00      | 1.00   | 1.00     | 36      |
| **Weighted avg** | 1.00      | 1.00   | 1.00     | 36      |


Teraz struktura modelu wygląda tak:
- warstwa wejściowa
- dense (modyfikowalna, 8/16/32 neuronów, opcjonalna normalizacja i regularyzacja), relu
- dense (wyjściowa): 3 neurony, softmax

Przetestowałem następujące kombinacje:

```python
configs = [
        {"hidden": 16, "tag": "none"},
        {"hidden": 16, "tag": "batchnorm_baseline"},
        {"hidden": 16, "tag": "l1"},
        {"hidden": 16, "tag": "l1_batchnorm"},
        {"hidden": 16, "tag": "l2"},
        {"hidden": 16, "tag": "l2_batchnorm"},
        {"hidden": 8, "tag": "l1_batchnorm"},
        {"hidden": 32, "tag": "l2_batchnorm"},
    ]
```
### Wnioski
Wyżej wspomniałem, że najmniejszy model osiągał 100% dokładności, jednak nie okazało się to regułą. Przy wielu próbach zdarzało się, że model nie dobił do 100% dokładności, ale mimo wszystko jest to najmniejszy model, który dawał niezłe wyniki. Pozostałe modele wahały się od 94 do 100% dokładności, a te, które regularnie osiągały 100% to te z 16 i 32 neuronami, reularyzacją l2 i batch normalization.

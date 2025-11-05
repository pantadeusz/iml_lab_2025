# Podusmowanie
## Zapis modeli 
Zapisane modele są nazwane w formacie 
```
{liczba epok}_{maksymalna liczba prób}_{liczba wykonań na test}_{numer testu}
```
To samo tyczy się tuningu, jednak dla poszczególnych wartości,
tuningi są nadpisywane.
## Wyniki
### Architektura sieci neuronowej:
Tuner zawsze wybierał pomiędzy `relu` i `tanh`, ale
warstwa wyjściowa zawsze była aktywowana jako
`sigmoid`.

Wszystkie modele miały warstwy 1 ukrytą, wyjściową
i w zależności od traila od 1 do 3 warstw pomiędzy nimi.

Rozmiary tych warstw wynosiły od 32 do 512.

### opis
Wyniki pokazują, że każdy model sieci neuronowej
używający tunera osiąga lepsze rezultaty niż 
model RFC. Niezależnie od wartości tunera model
osiągał precyzję ważoną na poziomie 94%-95%.
Eksperymenty pokazały, że w dostosowaniu tunera 
najwieksze znacznie ma `executions_per_trial`,
jednak niesie się to z wysokim kosztem procesowania
oraz czasem wykonania.

### Tabel wartości testowych Tunera:

| epoki | max_trials | executions_per_trial |
|-------|------------|----------------------|
| 30    | 2          | 5                    |
| 30    | 5          | 2                    |
| 30    | 5          | 10                   |
| 30    | 8          | 3                    |
| 50    | 5          | 2                    |


### WYNIKI OSTATNIEGO TESTU

#### Model Random Forest Classifier
**Classification Report**
```json
{
   "0":{
      "precision":0.9114077669902912,
      "recall":0.93875,
      "f1-score":0.9248768472906403,
      "support":800.0
   },
   "1":{
      "precision":0.9149305555555556,
      "recall":0.8783333333333333,
      "f1-score":0.8962585034013606,
      "support":600.0
   },
   "accuracy":0.9128571428571428,
   "macro avg":{
      "precision":0.9131691612729234,
      "recall":0.9085416666666666,
      "f1-score":0.9105676753460005,
      "support":1400.0
   },
   "weighted avg":{
      "precision":0.9129175335182618,
      "recall":0.9128571428571428,
      "f1-score":0.9126118427666633,
      "support":1400.0
   }
}
```
**Confusion Matrix:**

| **751** | **49**  |
|---------|---------|
| **73**  | **527** |


#### Best Neural Network Model
**Classification Report** 
```json
{
   "0":{
      "precision":0.96,
      "recall":0.96,
      "f1-score":0.96,
      "support":800.0
   },
   "1":{
      "precision":0.9466666666666667,
      "recall":0.9466666666666667,
      "f1-score":0.9466666666666667,
      "support":600.0
   },
   "accuracy":0.9542857142857143,
   "macro avg":{
      "precision":0.9533333333333334,
      "recall":0.9533333333333334,
      "f1-score":0.9533333333333334,
      "support":1400.0
   },
   "weighted avg":{
      "precision":0.9542857142857143,
      "recall":0.9542857142857143,
      "f1-score":0.9542857142857143,
      "support":1400.0
   }
}
```
**Confusion Matrix:**

| **768** | **32**  |
|---------|---------|
| **32**  | **568** |

### Wnioski
Tuner to naprawdę potężne narzędzie, dzięki któremu
udało się podnieść precyję na naprawdę wysoki poziom
ponad 95% z poniżej 92%. Jednak nie sądzę, żeby
był to dobry wybór dla każdego zbioru danych.
W danych `ITI Student Dropout Synthetic Dataset` wynik
na poziomie 92% precyzji jest w pełni wystarczający,
jednak gdyby był to zbiór danych medycznych, warto by
było poświęcić czas i złożoności dla tych kilku
punktów procentowych.
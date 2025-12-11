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


### Struktura najlepszego modelu

#### Best Neural Network Model

| Layer (type)  | Output Shape | Param # |
|---------------|--------------|---------|
| Tanh          | (None, 416)  | 19,552  |
| Tanh          | (None, 352   | 146,784 |
| Dropout(0.25) | (None,352)   | 0       |
| Sigmoid       | (None,1)     | 353     |

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
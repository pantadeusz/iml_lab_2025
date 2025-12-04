### A. Baseline Model (Trenowany i testowany na czystych danych)

Model MLP osiągnął wysoki wynik bazowy.

```
Loss: 0.3237, Accuracy: 0.9146
              precision    recall  f1-score   support

           0       0.95      0.99      0.97       980
           1       0.97      0.98      0.97      1135
           2       0.92      0.86      0.89      1032
           3       0.87      0.92      0.90      1010
           4       0.90      0.92      0.91       982
           5       0.92      0.84      0.87       892
           6       0.95      0.93      0.94       958
           7       0.93      0.93      0.93      1028
           8       0.83      0.90      0.86       974
           9       0.92      0.87      0.90      1009

    accuracy                           0.91     10000
   macro avg       0.91      0.91      0.91     10000
weighted avg       0.92      0.91      0.91     10000
```

### Zastosowana Augmentacja
Do testowania odporności modeli oraz treningu na trudnych danych wykorzystano następujące parametry augmentacji:

```
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.05),
    tf.keras.layers.RandomInvert(factor=0.04)
])
```

### B. Baseline Model (Ewaluacja na danych zaugmentowanych)

Drastyczny spadek skuteczności przy testowaniu na trudnych danych. Loss wzrósł do poziomu 960.

```
Loss: 960.7319, Accuracy: 0.5169
              precision    recall  f1-score   support

           0       0.87      0.52      0.65       980
           1       0.86      0.58      0.69      1135
           2       0.45      0.48      0.47      1032
           3       0.33      0.70      0.45      1010
           4       0.61      0.39      0.48       982
           5       0.45      0.43      0.44       892
           6       0.55      0.64      0.59       958
           7       0.62      0.46      0.53      1028
           8       0.48      0.36      0.41       974
           9       0.46      0.61      0.52      1009

    accuracy                           0.52     10000
   macro avg       0.57      0.52      0.52     10000
weighted avg       0.57      0.52      0.53     10000
```

### C. Baseline Model (Trenowany na danych zaugmentowanych)

Niewielka poprawa po treningu na trudnych danych, ale model nadal nie radzi sobie dobrze.

```
Loss: 31.9458, Accuracy: 0.6273
              precision    recall  f1-score   support

           0       0.76      0.79      0.77       980
           1       0.76      0.81      0.78      1135
           2       0.58      0.56      0.57      1032
           3       0.62      0.66      0.64      1010
           4       0.59      0.58      0.58       982
           5       0.55      0.47      0.50       892
           6       0.67      0.64      0.65       958
           7       0.51      0.71      0.59      1028
           8       0.66      0.44      0.53       974
           9       0.64      0.60      0.62      1009

    accuracy                           0.63     10000
   macro avg       0.63      0.63      0.63     10000
weighted avg       0.63      0.63      0.63     10000
```

### D. Model CNN (Trenowany na danych zaugmentowanych)

Najlepszy wynik. Model konwolucyjny "odzyskał" wysoką dokładność mimo trudnych danych.

```
Loss: 0.4403, Accuracy: 0.9222
              precision    recall  f1-score   support

           0       0.81      0.98      0.89       980
           1       0.92      0.98      0.95      1135
           2       0.95      0.92      0.94      1032
           3       0.91      0.95      0.93      1010
           4       0.95      0.92      0.93       982
           5       0.87      0.97      0.91       892
           6       0.99      0.91      0.95       958
           7       0.94      0.89      0.92      1028
           8       0.99      0.78      0.87       974
           9       0.92      0.90      0.91      1009

    accuracy                           0.92     10000
   macro avg       0.93      0.92      0.92     10000
weighted avg       0.93      0.92      0.92     10000
```

-----

## 2\. Tabela porównawcza

| Model | Architektura | Dane Treningowe | Dane Testowe | Accuracy | Loss | Rozmiar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | MLP (Dense) | Czyste | Czyste | **91.46%** | 0.32 | 1.2 MB |
| **Baseline** | MLP (Dense) | Czyste | Augmentowane | **51.69%** | 960.73 | 1.2 MB |
| **Baseline** | MLP (Dense) | Augmentowane | Augmentowane | **62.73%** | 31.95 | 1.2 MB |
| **CNN** | Conv2D | Augmentowane | Augmentowane | **92.22%** | 0.44 | 2.7 MB |

## 3\. Wnioski

1.  Prosta sieć neuronowa (Baseline) jest bardzo wrażliwa na zmiany geometryczne.
2.  Model konwolucyjny (CNN) poradził sobie z zadaniem znakomicie, osiągając wynik **92.22%** na trudnych danych. Jest to wynik lepszy niż bazowy model na czystych danych.
3. Do analizy obrazów, które mogą być przesunięte, obrócone lub zaszumione (augmentacja), niezbędne jest stosowanie warstw konwolucyjnych i poolingu, które skutecznie ekstrahują cechy niezależnie od ich położenia.
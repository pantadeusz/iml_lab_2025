# Główne

Wybrałem dla zrobienia zadania TensorFlow Datasets Keras Example

## Wnioski:

* Rozmiar modeli:
  * Baseline: 1.2 MB
  * Z augmentacją: 1.2 MB
  * Z warstwą conv + maxpool: 8.4 MB

1. Wyniki ewaluacji modelu bazowego oraz modelu z warstwami konwolucyjnymi i Max Pooling są do siebie zbliżone
2. Wyniki modelu bazowego testowanego na zbiorze po augmentacji, jak i modelu trenowanego na takich danych, są znacznie niższe od wyników wcześniej opisanych modeli
    
    -> Model wykorzystujący warstwy konwolucyjne oraz MaxPooling wydaje się być bardziej odpowiedni do praktycznych zastosowań. Osiąga wyniki zbliżone do modelu bazowego + trenowany był na danych z augmentacją, co oznacza że model ten jest bardziej odporny na drobne zniekształcenia, przesunięcia czy obroty obiektów, co robi go modelem prioretytowym 

### Step 0/1
- Rozmiar batchu = 128
- Liczba epoch uczenia - 6
- Jedna warstwa ReLu - 128
- Learning rate = 0.001
- Optymilizator - Adam

Wyniki baseline modeli na bazowym zbiorze treningowym:
```
                          precision    recall  f1-score   support

           0       0.96      0.97      0.97       980
           1       0.97      0.97      0.97      1135
           2       0.93      0.90      0.92      1032
           3       0.89      0.93      0.91      1010
           4       0.94      0.92      0.93       982
           5       0.89      0.87      0.88       892
           6       0.94      0.95      0.94       958
           7       0.92      0.93      0.93      1028
           8       0.89      0.85      0.87       974
           9       0.88      0.92      0.90      1009

    accuracy                           0.92     10000
   macro avg       0.92      0.92      0.92     10000
weighted avg       0.92      0.92      0.92     10000
```

### Step 2/3 - augmentacja
Wykorzystałem taki parametry dla zmiany zdjęć z zbiorze testowym:
```
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
    tf.keras.layers.RandomInvert(factor=0.2)
])
```

Wyniki baseline modeli na zbiorze po augmentacji:
```
                          precision    recall  f1-score   support

           0       0.88      0.49      0.63       980
           1       0.93      0.42      0.58      1135
           2       0.63      0.63      0.63      1032
           3       0.34      0.81      0.48      1010
           4       0.78      0.52      0.63       982
           5       0.49      0.58      0.53       892
           6       0.67      0.64      0.65       958
           7       0.65      0.67      0.66      1028
           8       0.51      0.42      0.46       974
           9       0.61      0.60      0.60      1009

    accuracy                           0.58     10000
   macro avg       0.65      0.58      0.59     10000
weighted avg       0.65      0.58      0.59     10000
```

### Step 4

Wyniki augmentation modeli na zbiorze po augmentacji:
```
                          precision    recall  f1-score   support

           0       0.77      0.74      0.75       980
           1       0.68      0.74      0.71      1135
           2       0.65      0.51      0.57      1032
           3       0.68      0.62      0.65      1010
           4       0.65      0.57      0.60       982
           5       0.61      0.52      0.56       892
           6       0.64      0.67      0.65       958
           7       0.77      0.65      0.70      1028
           8       0.72      0.47      0.57       974
           9       0.37      0.72      0.49      1009

    accuracy                           0.62     10000
   macro avg       0.65      0.62      0.63     10000
weighted avg       0.65      0.62      0.63     10000
```

### Step 5/6

Dodałem po jednej warstwie conwulucyjnej i max pooling do naszej modeli 

Otrzymałem takie wyniki ewaluacji:
```
                          precision    recall  f1-score   support

           0       0.95      0.89      0.91       980
           1       0.57      0.99      0.72      1135
           2       0.93      0.85      0.89      1032
           3       0.98      0.82      0.89      1010
           4       0.97      0.87      0.92       982
           5       0.96      0.86      0.91       892
           6       0.97      0.87      0.92       958
           7       0.95      0.83      0.89      1028
           8       0.89      0.87      0.88       974
           9       0.93      0.85      0.88      1009

    accuracy                           0.87     10000
   macro avg       0.91      0.87      0.88     10000
weighted avg       0.90      0.87      0.88     10000
```
# Wnioski

## Opis architektury modelu DNN (baseline)

- Warstwa wejściowa: 30 cech
- Warstwa ukryta 1: Dense(16, activation='relu')
- Warstwa ukryta 2: Dense(8, activation='relu')
- Warstwa wyjściowa: Dense(1, activation='sigmoid') 

#### Hiperparametry

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: binary_crossentropy
- Batch size: 32
- Epochs: 20

## Opis eksperymentu 

Eksperyment miał na celu dostosowanie modelu DNN tak, aby uzyskał lepszy wynik niż rozwiązanie oparte na modelu Random Forest z biblioteki scikit-learn.

W celu dostosowania modelu DNN użyty został Keras Tuner.

Zadaniem była klasyfikacja binarna (nowotwór złośliwy vs łagodny). Za dataset posłużył mi zbiór **Breast Cancer Wisconsin** pobierany z scikit-learn. Użyty został również StandardScaler w celu standaryzacji cech.

### Wyniki modeli 

#### RandomForest

- Validation Accuracy: 0.9670

| Class        | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| 0           | 0.98      | 0.93   | 0.95     | 43      |
| 1           | 0.96      | 0.99   | 0.97     | 71      |
| **Accuracy**|           |        | **0.96** | 114     |
| Macro avg   | 0.97      | 0.96   | 0.96     | 114     |
| Weighted avg| 0.97      | 0.96   | 0.96     | 114     |

#### DNN (baseline)
- Validation Accuracy - 0.9560

| Class        | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| 0           | 0.93      | 0.95   | 0.94     | 43      |
| 1           | 0.97      | 0.96   | 0.96     | 71      |
| **Accuracy**|           |        | **0.96** | 114     |
| Macro avg   | 0.95      | 0.96   | 0.95     | 114     |
| Weighted avg| 0.96      | 0.96   | 0.96     | 114     |

## Eksperymenty z autotunerem

### Eksperyment 1

#### Parametry autotunera

- Tuner: RandomSearch
- Cel: val_accuracy
- Liczba prób: 10
- Liczba epok: 20

#### Tunowane hiperparametry
- units_1: 8–64 (step 8)
- dropout_1: 0.0–0.3 (step 0.1)
- units_2: 4–32 (step 4)
- dropout_2: 0.0–0.3 (step 0.1)
- learning_rate: 0.0001–0.01 (step 0.001)

#### Najlepsze hiperparametry znalezione przez tuner

- units_1: 48 (było: 16)
- units_2: 16 (było: 8)
- dropout_1: 0.00 (było: 0.0)
- dropout_2: 0.00 (było: 0.0)
- learning_rate: 0.008100 (było: 0.001)

#### Wyniki: 

- Validation Accuracy: 0.9670

| Class        | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| 0           | 0.95      | 0.98   | 0.97     | 43      |
| 1           | 0.99      | 0.97   | 0.98     | 71      |
| **Accuracy**|           |        | **0.97** | 114     |
| Macro avg   | 0.97      | 0.97   | 0.97     | 114     |
| Weighted avg| 0.97      | 0.97   | 0.97     | 114     |

### Eksperyment 2

#### Parametry autotunera
- Tuner: RandomSearch
- Cel: val_accuracy
- Liczba prób: 20
- Liczba epok: 20

#### Tunowane hiperparametry
- units_1: 32–128 (step 16)
- dropout_1: 0.0–0.5 (step 0.1)
- units_2: 8–64 (step 8)
- dropout_2: 0.0–0.5 (step 0.1)
- learning_rate: 0.007–0.01 (step 0.001)

#### Najlepsze hiperparametry znalezione przez tuner
- units_1: 128 (było: 16)
- units_2: 64 (było: 8)
- dropout_1: 0.30 (było: 0.0)
- dropout_2: 0.00 (było: 0.0)
- learning_rate: 0.010000 (było: 0.001)

#### Wyniki:
- Validation Accuracy: 0.9780

| Class        | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| 0           | 0.98      | 0.95   | 0.96     | 43      |
| 1           | 0.97      | 0.99   | 0.98     | 71      |
| **Accuracy**|           |        | **0.97** | 114     |
| Macro avg   | 0.97      | 0.97   | 0.97     | 114     |
| Weighted avg| 0.97      | 0.97   | 0.97     | 114     |


## Wnioski końcowe
Eksperymenty wykazały, że Keras Tuner poprawił wyniki i model DNN z automatycznie dobranymi hiperparametrami spisał się lepiej, niż model Random Forest i ręcznie skonfigurowany DNN.

Pomimo już dobrych wyników modeli bazowych (RF i DNN) dzięki zwiększeniu liczby neuronów w obu warstwach oraz zwiększeniu learning rate uzyskaliśmy wzrost accuracy na zbiorze testowym 96% -> 97%.

Oba eksperymenty z tunerem dały identyczne wyniki na zbiorze testowym (3 błędy, 97% accuracy), a dalsze zwiększenie zakresów hiperparametrów nie przyniosło pożądanych rezultatów.
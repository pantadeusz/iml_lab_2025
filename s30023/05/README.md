# Wnioski
## Architektura
- Warstwa wejściowa: 4 cechy 
- Warstwa ukryta: Jedna warstwa gęsta (Dense) z aktywacją ReLU. Liczba neuronów była parametrem tuner-a
- Warstwa wyjściowa: Warstwa Dense z 3 neuronami i aktywacją Softmax

## Opis eksperypmentru
- Celem eksperymentu było automatyczne znalezienie optymalnych hiperparametrów dla naszego modela neuronowego. Użyłem do tego Keras Tuner
- Porównano wydajność modelu bazowego (ręcznie ustawione parametry) z modelem znalezionym przez tuner

## Paramentry tunera
- lr (learning rate) - 0.001, 0.01, 0.1
- units (liczba neuronów warstwy ukrytej) - od 32 do 512 z krokiem 32
- Cel: val_accuracy (maksymalizacja dokładności na zbiorze walidacyjnym)

## Wynniki eksperymetru
- Krótko: Autotune hiperparametrów poprawił metryki
- Model bazowy ma accuracy 90%, ale na classification_report widać że ma niskie recall [0.67] dla klasy versicolor oraz niską precision [0.79] dla virginica -> model ma problem z rozróżnianiem tych dwóch klas 
- Tuner odkrył, że zwiększenie liczby neuronów oraz zwiększenia learning rate poprawia sytuację. Zoptymalizowany model ma wyższą accuracy (97%) i też poprawił metryki dla versicolor i virginica 

# Wynik DO eksperymentu z Tuner 
- Units in hidden layer: 64
- Learning rate: 0.001

```
                      precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.67      0.80         9
   virginica       0.79      1.00      0.88        11

    accuracy                           0.90        30
   macro avg       0.93      0.89      0.89        30
weighted avg       0.92      0.90      0.90        30
```

# Wynik PO eksperymentach z Tuner 
- Próba 1:

### Best hyperparameters:
- Units in hidden layer: 416
- Learning rate: 0.01
### EVALUATION
```
                      precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.89      0.94         9
   virginica       0.92      1.00      0.96        11

    accuracy                           0.97        30
   macro avg       0.97      0.96      0.97        30
weighted avg       0.97      0.97      0.97        30
```

- Próba 2:

### Best hyperparameters:
- Units in hidden layer: 192
- Learning rate: 0.01
### EVALUATION
```
                      precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.75      1.00      0.86         9
   virginica       1.00      0.73      0.84        11

    accuracy                           0.90        30
   macro avg       0.92      0.91      0.90        30
weighted avg       0.93      0.90      0.90        30
```

- Próba 3:

### Best hyperparameters:
- Units in hidden layer: 512
- Learning rate: 0.01
### EVALUATION
```
                      precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.89      0.94         9
   virginica       0.92      1.00      0.96        11

    accuracy                           0.97        30
   macro avg       0.97      0.96      0.97        30
weighted avg       0.97      0.97      0.97        30
```
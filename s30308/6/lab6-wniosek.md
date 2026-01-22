# Podsumowanie wyników modelu Beans

## Najlepsze hiperparametry znalezione przez Keras Tuner

| Hyperparameter  | Najlepsza wartość |
|-----------------|-----------------|
| `units_1`       | 256             |
| `activation_1`  | tanh            |
| `units_2`       | 224             |
| `activation_2`  | tanh            |
| `optimizer`     | sgd             |

---

## Wyniki treningu i ewaluacji

- Dokładność na zbiorze testowym: **~70%**
- F1-score dla poszczególnych klas:
  - `angular_leaf_spot`: 0.75  
  - `bean_rust`: 0.68  
  - `healthy`: 0.68
- Model lepiej klasyfikuje `angular_leaf_spot`, gorzej radzi sobie z `bean_rust` i `healthy`.

---

## Wnioski

1. Dodanie Keras Tuner nie poprawiło znacząco dokładności modelu — najwyraźniej prosty Feedforward Neural Network nie jest wystarczająco wydajny na obrazach.  
2. Największy ogranicznik to struktura modelu — obrazy najlepiej przetwarzają **Convolutional Neural Networks (CNN)**.  
3. Obecny skrypt do klasyfikacji pojedynczego obrazu działa poprawnie i umożliwia szybkie sprawdzenie predykcji na przykładzie obrazu.

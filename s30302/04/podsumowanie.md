# Porównanie modeli: Random Forest vs DNN

## Dane
Użyty zbiór danych: **Breast Cancer Wisconsin (Diagnostic)** z `sklearn.datasets`.

Zadanie: klasyfikacja (czy guz jest złośliwy czy łagodny).

## Modele
- **Random Forest Classifier** (`sklearn.ensemble`)
- **Deep Neural Network (DNN)** zbudowany w **TensorFlow/Keras**

## Wyniki
| Model              | Dokładność (accuracy) |
|--------------------|-----------------------|
| Random Forest      | 0.9649                |
| DNN (TensorFlow)   | 0.9737                |

## Wnioski
Oba modele osiągnęły bardzo wysoką skuteczność, powyżej 96%.  
Model DNN uzyskał nieco lepszy wynik, jednak różnica jest niewielka.  
Random Forest uczy się szybciej i wymaga mniej konfiguracji,  
natomiast DNN daje większe możliwości rozbudowy i eksperymentów (np. z liczbą warstw czy neuronów).
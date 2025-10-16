# Macierz pomyłek 
1. True Positive (TP) - Model poprawnie przewidział klasę pozytywną
2. False Positive (FP) - Model błędnie przewidział klasę pozytywną, gdy w rzeczywistości była negatywna
3. False Negative (FN) - Model błędnie przewidział klasę negatywną, gdy w rzeczywistości była pozytywna
4. True Negative (TN) - Model poprawnie przewidział klasę negatywną

# Miary klasyfikacji
'1' i '0' w niej to etykity dla klas:
- 1 - etykieta dla klasy 1 (pozytywnej - TP i FP)
- 0 - etykieta dla klasy 0 (negatywnej - TN i FN)

Dla 0:
- precision [albo Specifity] = TN / (TN + FN) 
  - Odsetek poprawnie wykrytych przypadków negatywnych.
- recall [albo Negative Predictive Valeu] = TN / (TN + FP)
  - Odsetek prawdziwych negatywnych wśród wykrytych negatywnych.

Dla 1:
- precision = TP / (TP + FP). 
  - Odsetek prawdziwych pozytywnych wśród wszystkich wykrytych pozytywnych.
- recall [albo Sensitivity] = TP / (TP + FN). 
  - odsetek poprawnie wykrytych przypadków pozytywnych.


Dla wszystkich klas:
- f1-score: 2 * precision * recall / (precision + recall)
  - Średnia harmoniczna precyzji i czułości.
- accuracy: (TP + TN) / (TP + TN + FP + FN)
  - Odsetek poprawnych predykcji ogółem.
- support - ile elementów należy do tej klasy w rzeczywistości

Macro avg + Weighted avg:
- macro avg: prosta średnia z metryk dla wszystkich klas
- weighted avg: średnia ważona liczbą przykładów

# Próg decyzyjny 

- Niski próg -> wysoka czułość, niska swoistość
- oki próg -> wyższa swoistość, niższa czułość
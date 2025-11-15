## 🌳 Model Bazowy: Random Forest
> **Parametry:** `n_estimators=50`, `criterion='log_loss'`

### Raport Klasyfikacji (Random Forest)
```py
              precision    recall  f1-score   support

           0       0.92      0.87      0.89        82
           1       0.90      0.94      0.92       102

    accuracy                           0.91       184
   macro avg       0.91      0.90      0.91       184
weighted avg       0.91      0.91      0.91       184
```

## 🧠 Sieć Neuronowa (Tuned)
Model sieci neuronowej został znaleziony i zoptymalizowany przy użyciu Keras Tuner metodą random search.

### 🛠️ Architektura i Trening
* **Model:** Sieć neuronowa dwuwarstwowa
* **Optymalizator:** `Adam`
* **Funkcja kosztu:** Binarna entropia krzyżowa
* **Trening:** Użycie batchy o rozmiarze 32

### ⚙️ Strojenie Hiperparametrów
* **Liczba prób:** Wypróbowano 50 różnych kombinacji modeli.
* **Epoki:** Każdy model był trenowany przez maksymalnie 150 epok.
* **Wczesne zatrzymanie:** Zastosowano `EarlyStopping` z `patience=10`. Trening był przerywany, jeśli dokładność walidacyjna nie poprawiła się przez 10 kolejnych epok.

### Najlepsze znalezione hiperparametry:
* **Layer 1 Units:** 40
* **Layer 2 Units:** 12
* **Learning Rate:** ~0.00051

### Raport Klasyfikacji (Sieć Neuronowa)
```py
              precision    recall  f1-score   support

           0     0.9125    0.8902    0.9012        82
           1     0.9135    0.9314    0.9223       102

    accuracy                         0.9130       184
   macro avg     0.9130    0.9108    0.9118       184
weighted avg     0.9130    0.9130    0.9129       184
```

## 📈 Wnioski
> Modelowi sieci neuronowej udało się uzyskać nieznacznie większą dokładność na zbiorze walidacyjnym niż modelowi Random Forest.
> * **Sieć Neuronowa (Tuned):** ~0.9130
> * **Random Forest (Baseline):** ~0.9076
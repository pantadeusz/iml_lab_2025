# Lab 5: Strojenie hiperparametrów z użyciem Keras Tuner

## 1. Architektura modelu

### Model bazowy Scikit-Learn
- **Algorytm:** Regresja logistyczna  
- **Parametry:** max_iter=3000, random_state=42  
- **Przetwarzanie wstępne:** Normalizacja przy użyciu StandardScaler

### Architektura głębokiej sieci neuronowej (DNN)

#### Bazowy model DNN (bez strojenia)
- **Warstwa wejściowa:** 30 cech  
- **Warstwa ukryta 1:** 64 neurony, aktywacja ReLU, Dropout(0.3)  
- **Warstwa ukryta 2:** 32 neurony, aktywacja ReLU, Dropout(0.2)  
- **Warstwa wyjściowa:** 1 neuron, aktywacja Sigmoid  
- **Optymalizator:** Adam (domyślny learning rate)  
- **Funkcja straty:** Binary Crossentropy  
- **Uczenie:** 50 epok, batch_size = 32

#### Strojoną DNN (z Keras Tuner)
- **Warstwa wejściowa:** 30 cech  
- **Warstwa ukryta 1:** Regulowana (32–256 neuronów), aktywacja ReLU, Dropout regulowany (0.0–0.5)  
- **Warstwa ukryta 2:** Regulowana (16–128 neuronów), aktywacja ReLU, Dropout regulowany (0.0–0.5)  
- **Warstwa wyjściowa:** 1 neuron, aktywacja Sigmoid  
- **Optymalizator:** Adam z regulowanym współczynnikiem uczenia  
- **Funkcja straty:** Binary Crossentropy  
- **Uczenie:** 30 epok na próbę, batch_size = 32

---

## 2. Opis eksperymentu

Celem eksperymentu było porównanie trzech podejść do klasyfikacji raka piersi:

1. **Model bazowy Scikit-Learn:** klasyczna regresja logistyczna jako punkt odniesienia  
2. **Bazowy DNN:** sieć neuronowa o stałych hiperparametrach  
3. **Strojona DNN:** sieć neuronowa z hiperparametrami optymalizowanymi przy użyciu Keras Tuner

Celem było sprawdzenie, czy strojenie hiperparametrów może poprawić wyniki DNN w porównaniu z klasycznym modelem uczenia maszynowego.

### Podział danych:
- **Zbiór treningowy:** 364 próbki (80% z 455 próbek po początkowym podziale)  
- **Zbiór walidacyjny:** 91 próbek (20% z 455, użyty do strojenia)  
- **Zbiór testowy:** 114 próbek (całkowicie odrębny)

---

## 3. Parametry Keras Tuner

### Konfiguracja tunera:
- **Typ tunera:** RandomSearch  
- **Cel:** Dokładność walidacyjna (val_accuracy)  
- **Maksymalna liczba prób:** 10  
- **Liczba wykonanych trenowań na próbę:** 1  
- **Łączny czas trenowania:** ~30 sekund (ok. 0,30 minuty)  
- **Średni czas jednej próby:** ~1,8 sekundy

### Strojenie hiperparametrów:
1. **units_layer_1:** liczba neuronów w pierwszej warstwie ukrytej  
   - Zakres: 32–256 (krok: 32)  
   - Wartości: [32, 64, 96, 128, 160, 192, 224, 256]

2. **dropout_1:** współczynnik dropout po pierwszej warstwie ukrytej  
   - Zakres: 0.0–0.5 (krok: 0.1)  
   - Wartości: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

3. **units_layer_2:** liczba neuronów w drugiej warstwie ukrytej  
   - Zakres: 16–128 (krok: 16)  
   - Wartości: [16, 32, 48, 64, 80, 96, 112, 128]

4. **dropout_2:** współczynnik dropout po drugiej warstwie ukrytej  
   - Zakres: 0.0–0.5 (krok: 0.1)  
   - Wartości: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

5. **learning_rate:** współczynnik uczenia optymalizatora Adam  
   - Wartości: [0.01, 0.001, 0.0001]

### Najlepsze znalezione hiperparametry:
- **Warstwa 1 – liczba neuronów:** 128  
- **Dropout 1:** 0.3  
- **Warstwa 2 – liczba neuronów:** 32  
- **Dropout 2:** 0.0  
- **Współczynnik uczenia:** 0.01

---

## 4. Wyniki

### Model bazowy Scikit-Learn (Regresja logistyczna)

**Macierz pomyłek:**
```
                      Pred 0  Pred 1
Actual 0 (złośliwy)        41       1
Actual 1 (łagodny)          1      71
```

**Raport klasyfikacji:**
```
              precision    recall  f1-score   support
   złośliwy       0.98      0.98      0.98        42
      łagodny     0.99      0.99      0.99        72
    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114
```

---

### Bazowy DNN (bez strojenia)

**Macierz pomyłek:**
```
                      Pred 0  Pred 1
Actual 0 (złośliwy)        41       1
Actual 1 (łagodny)          2      70
```

**Raport klasyfikacji:**
```
              precision    recall  f1-score   support
   złośliwy       0.95      0.98      0.96        42
      łagodny     0.99      0.97      0.98        72
    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
```

---

### DNN po strojeniu (z Keras Tuner)

**Macierz pomyłek:**
```
                      Pred 0  Pred 1
Actual 0 (złośliwy)        40       2
Actual 1 (łagodny)          4      68
```

**Raport klasyfikacji:**
```
              precision    recall  f1-score   support
   złośliwy       0.91      0.95      0.93        42
      łagodny     0.97      0.94      0.96        72
    accuracy                           0.95       114
   macro avg       0.94      0.95      0.94       114
weighted avg       0.95      0.95      0.95       114
```

---

### Porównanie wyników

| Model                     | Dokładność | Precyzja (ważona) | Czułość (ważona) | F1 (ważony) |
|----------------------------|-------------|--------------------|------------------|--------------|
| Scikit-Learn (bazowy)      | 0.982       | 0.982              | 0.982            | 0.982        |
| DNN (bazowy)               | 0.974       | 0.974              | 0.974            | 0.974        |
| DNN (po strojeniu)         | 0.947       | 0.948              | 0.947            | 0.948        |

---

## 5. Wnioski

Eksperyment doprowadził do kilku istotnych obserwacji:

**Nieoczekiwane wyniki:**  
Wbrew oczekiwaniom, strojony model DNN (95% dokładności) osiągnął gorszy wynik niż zarówno bazowa regresja logistyczna (98%), jak i bazowy DNN (97%). Jest to jednak **„kontrolowana porażka”**, z której można wyciągnąć cenne wnioski.

**Najważniejsze spostrzeżenia:**

1. **Przewaga prostego modelu:** Klasyczna regresja logistyczna uzyskała najlepsze wyniki, co pokazuje, że proste modele mogą być bardzo skuteczne w przypadku dobrze ustrukturyzowanych danych tabelarycznych, takich jak Breast Cancer Dataset.  
2. **Ograniczony zakres strojenia:** 10 prób to zbyt mało, aby skutecznie przeszukać przestrzeń hiperparametrów — tuner mógł wybrać zbyt wysoki learning rate (0.01), co spowodowało niestabilność treningu.  
3. **Ryzyko przeuczenia:** Spadek dokładności modelu strojonego może wynikać z przeuczenia na zbiorze walidacyjnym lub zbyt agresywnych parametrów.  
4. **Charakterystyka danych:** Dane są stosunkowo proste (30 cech, wyraźna liniowa separowalność), co powoduje, że sieci neuronowe są w tym przypadku zbyt złożone.

**Wnioski końcowe:**  
Eksperyment pokazuje, że bardziej złożony model i strojenie hiperparametrów nie zawsze prowadzą do lepszych wyników. Rzetelne udokumentowanie takiego przypadku jest równie cenne — pokazuje znaczenie:  
(a) wystarczających zasobów obliczeniowych,  
(b) zrozumienia danych przed doborem architektury modelu,  
(c) wykorzystywania prostszych modeli jako punktu odniesienia.  

W praktyce produkcyjnej dla tego zbioru danych **regresja logistyczna** byłaby najlepszym wyborem ze względu na najwyższą skuteczność, interpretowalność i efektywność obliczeniową.

# Laboratorium 7

## 1. Wstęp
Celem laboratorium było zbudowanie modelu klasyfikacyjnego dla zbioru danych **Wine** z UCI.  
Testowano zarówno klasyczny **RandomForestClassifier**, jak i sieci neuronowe z różnymi konfiguracjami warstw.  

W laboratorium skupiono się na:
- Normalizacji danych wejściowych,
- Budowie sieci neuronowych o różnych rozmiarach,
- Eksperymentach z redukcją liczby neuronów w celu zmniejszenia rozmiaru modelu przy zachowaniu 100% dokładności,
- Porównaniu wyników z klasycznym RandomForestClassifier.

---

## 2. Dane
- Liczba próbek: 36 w zbiorze testowym (po podziale na trening/test 80/20)
- Liczba cech: 13
- Liczba klas: 3

---

## 3. Sieć neuronowa - konfiguracja bazowa
- Model: 3 warstwy  
  - Warstwa 1: 128 neuronów, ReLU  
  - Warstwa 2: 64 neuronów, ReLU  
  - Warstwa 3: `num_classes` neuronów, Softmax  

- Funkcja straty: `sparse_categorical_crossentropy`  
- Optymalizator: Adam  

---

## 4. Testowane kombinacje sieci neuronowej

| Kombinacja | Liczba warstw | Neurony w warstwach | Epochs | Accuracy | Loss   |
|------------|---------------|-------------------|--------|---------|--------|
| 1          | 3             | 128-64-3          | 5      | 1.0     | 0.0659 |
| 2          | 3             | 64-32-3           | 5      | 1.0     | 0.3133 |
| 3          | 3             | 32-16-3           | 20     | 0.9722  | 0.1022 |
| 4          | 2             | 64-3              | 10     | 1.0     | 0.1992 |
| 5          | 2             | 32-3              | 10     | 1.0     | 0.3460 |
| 6          | 2             | 16-3              | 20     | 1.0     | 0.1768 |
| 7          | 2             | 8-3               | 40     | 0.9722  | 0.1860 |

---

## 5. Analiza i obserwacje

1. **Redukcja liczby neuronów**  
   - Modele sieci neuronowej osiągały 100% dokładności nawet przy znacznie zmniejszonej liczbie neuronów.  
   - Najmniejsza sieć zachowująca 100% dokładności miała konfigurację **16-3** (kombinacja 6) z 2 warstwami.

2. **Wpływ liczby warstw**  
   - Modele 3-warstwowe z większą liczbą neuronów szybciej osiągały wysoką dokładność.  
   - Modele 2-warstwowe wymagały więcej epok, ale mogły zachować 100% dokładności przy odpowiednim doborze neuronów.

3. **Porównanie z RandomForest**  
   - RandomForest osiągnął 100% dokładności, ale model sieci neuronowej daje większą elastyczność w regularyzacji i optymalizacji rozmiaru.

4. **Wnioski praktyczne**  
   - W przypadku małych zbiorów danych, liczba neuronów i warstw może być znacznie zredukowana bez utraty dokładności.  
   - Zmniejszenie liczby neuronów wpływa również na rozmiar zapisanego modelu `.keras`.  

---

## 6. Podsumowanie

- Sieci neuronowe z normalizacją danych oraz starannie dobraną liczbą neuronów mogą osiągać maksymalną dokładność przy minimalnym rozmiarze modelu.  
- Dobrze dobrana architektura (np. 16-3) pozwala zachować 100% accuracy przy znacznym zmniejszeniu liczby parametrów.  
- RandomForest sprawdził się równie dobrze w klasyfikacji tego zbioru danych, ale sieć neuronowa daje większe możliwości dalszej regularyzacji i eksperymentów z modelami głębokimi.  

> Laboratorium pokazuje, że w małych zbiorach danych warto eksperymentować z rozmiarem sieci, aby znaleźć optymalny balans między dokładnością a rozmiarem modelu.

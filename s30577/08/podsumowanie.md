# Raport: Lab 08 - Augmentacja Danych i Sieci CNN

Wszystkie modele były trenowane przy użyciu następujących, stałych parametrów:
* **Zbiór danych:** MNIST (Fashion MNIST / Digits)
* **Liczba epok:** 5
* **Rozmiar batcha:** 64
* **Algorytm uczenia:** Adam (Learning Rate = 0.001)
* **Augmentacja:** Losowy obrót (ok. 18°), przesunięcie (10%), losowy negatyw (20% szans).

---

## Tabela Wyników Zbiorczych

|       | Model (Architektura) | Dane Treningowe | Accuracy (Test Czysty) | Accuracy (Test Augmentowany) | Wnioski |
|:-----:|:---------------------|:----------------|:-----------------------:|:----------------------------:|:--------|
| **1** | Baseline (Dense) | Czyste | **0.9739** | **0.5592** | Model nie radzi sobie z najmniejszymi zmianami (brak generalizacji). |
| **2** | Baseline (Dense) | Augmentowane | **0.9636** | **0.9135** | Ogromna poprawa stabilności kosztem minimalnego spadku na danych idealnych. |
| **3** | CNN (Conv2D) | Augmentowane | **0.9872** | **0.9715** | **Najlepszy wynik.** Sieć uczy się kształtów, a nie pozycji pikseli. |

---

## 3. Podsumowanie
Przeprowadzone badanie pokazuje, że:
1.  **Sama architektura Dense (Baseline)** jest wrażliwa na położenie pikseli – wystarczy przesunąć cyfrę, by sieć przestała ją rozpoznawać.
2.  **Augmentacja danych** jest kluczowa w budowaniu odpornych modeli – pozwala "na siłę" nauczyć prostą sieć radzenia sobie z wariantami danych.
3.  **Architektura CNN** jest naturalnie przystosowana do pracy z obrazem. Dzięki operacjom splotu wykrywa cechy lokalne (krawędzie, łuki) niezależnie od ich umiejscowienia na obrazie, co w połączeniu z augmentacją daje model bliski perfekcji (97%+ na trudnych danych).
# Lab 08 – Podsumowanie

## 0. Parametry trenowania modeli
Wszystkie modele trenowano z następującymi parametrami:  
- **Liczba epok:** 10–30 (w zależności od modelu, EarlyStopping użyty)  
- **Batch size:** 32  
- **Optimizer:** Adam  
- **Learning rate:** domyślny lub lekko zmodyfikowany dla modeli custom  
- **Funkcja straty:** SparseCategoricalCrossentropy  
- **Metryki:** Accuracy  

---

## 1. Model baseline
- Model wytrenowany bez augmentacji danych.  
- Zapisany jako: `beans_model_final.h5`  
- Ewaluacja dowolnego modelu możliwa dzięki przygotowanej funkcji.  

**Wynik na czystym zbiorze testowym:**  
**Accuracy:** 0.789  

---

## 2. Mechanizm augmentacji danych
- Stworzono funkcję augmentacji obejmującą:  
  - losowy negatyw (odwracanie kolorów)  
  - losowy obrót ±10%  
  - losowe przesunięcie ±10%  
- Funkcjonalność sprawdzono wizualnie – działa poprawnie.  

---

## 3. Baseline testowany na danych z augmentacją
- Baseline model oceniono na „pomieszanych” danych testowych (augmentacja).  

**Wynik:**  
**Accuracy:** 0.453  

**Komentarz:** Model nie radzi sobie dobrze z przekształconymi danymi → brak odporności.  

---

## 4. Model trenowany na danych z augmentacją
- Model o tej samej architekturze co baseline, ale trenowany na danych wzbogaconych.  
- Zapisany jako: `beans_model_best.h5`  

**Wynik na czystym teście:**  
**Accuracy:** 0.789  

**Komentarz:** Stabilniejszy niż baseline, choć dokładność podobna na czystym zbiorze.  

---

## 5. Nowy model – Custom CNN
- Głębsza architektura:  
  - Kilka warstw Conv2D z MaxPooling  
  - Dropout  
  - Warstwy gęste na końcu  
- Wejście i wyjście zgodne z datasetem  

---

## 6. Trenowanie modelu custom i jego ocena
- Model wytrenowany z augmentacją danych.  

**Wynik:**  
**Accuracy:** 0.945  

Najlepszy wynik spośród wszystkich modeli.  

---

## 7. Porównanie modeli

| Model                     | Accuracy | Uwagi |
|---------------------------|---------|-------|
| **Baseline**              | 0.789   | Trening bez augmentacji – solidny, prosty |
| **Baseline na test_aug**  | 0.453   | Duży spadek jakości – brak odporności |
| **Model trenowany na augmentacji** | 0.789 | Stabilniejszy, wynik podobny do baseline |
| **Custom CNN**            | 0.945   | Najlepszy – głębsza architektura + augmentacja |

---

## 8. Wnioski końcowe
1. Augmentacja danych nie zwiększyła dokładności baseline, ale poprawiła jego odporność.  
2. Model trenowany na augmentowanych danych jest bardziej stabilny, choć wynik podobny do baseline.  
3. Największy wzrost jakości daje **bardziej złożona architektura** – Custom CNN osiągnął najwyższą dokładność.  
4. Łączenie augmentacji i głębszej sieci daje najlepszą generalizację i skuteczność.  

---
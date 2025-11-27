# Lab 8 - Raport: Augmentacja i CNN

## 1. Cel i przebieg
W ramach laboratorium przeprowadzono eksperyment sprawdzający wpływ **augmentacji danych** (losowe obroty, negatywy, przesunięcia) na jakość uczenia. Porównano również skuteczność klasycznej sieci warstwowej (`Dense`) z siecią konwolucyjną (`CNN`).

Eksperyment składał się z 4 etapów:
1.  Trening bazowy (Baseline) na czystych danych.
2.  **Stress Test:** Sprawdzenie modelu bazowego na danych zniekształconych.
3.  Trening zwykłej sieci (`Dense`) na danych z augmentacją.
4.  Trening sieci konwolucyjnej (`CNN`) na danych z augmentacją.

## 2. Wyniki eksperymentu

Poniższa tabela przedstawia uzyskane dokładności (Accuracy):

| Model | Dane treningowe | Dane testowe | Wynik (Accuracy) |
| :--- | :--- | :--- | :--- |
| **Baseline (Dense)** | Czyste | Czyste | **50.78%** |
| **Baseline (Stress Test)** | Czyste | Zniekształcone | **31.25%** |
| **Dense + Augmentacja** | Zniekształcone | Czyste/Zniekształcone | **33.59%** |
| **CNN + Augmentacja** | Zniekształcone | Czyste | **60.94%** |

## 3. Wnioski

1.  **Wrażliwość sieci Dense:** Model uczony na statycznych obrazkach zupełnie nie radzi sobie ze zmianami. W *Stress Teście* wynik spadł z ~51% do **31%**, co przy trzech klasach oznacza dokładność bliską losowemu zgadywaniu. Sieć nauczyła się układu pikseli, a nie rozpoznawania obiektów.

2.  **Porażka Dense przy augmentacji:** Próba nauczenia zwykłej sieci na trudnych, zróżnicowanych danych (obroty, negatywy) zakończyła się niepowodzeniem (**33.59%**). Architektura oparta tylko na warstwach gęstych jest zbyt prosta, by "zrozumieć" tak dużą zmienność danych (tzw. underfitting).

3.  **Przewaga CNN:** Sieć konwolucyjna okazała się bezkonkurencyjna. Mimo trenowania na trudnych danych, osiągnęła wynik **60.94%**. Dzięki warstwom konwolucyjnym i poolingowi, sieć potrafi wykrywać kluczowe cechy (kształty) niezależnie od ich położenia czy obrotu, co jest kluczowe w analizie obrazu.
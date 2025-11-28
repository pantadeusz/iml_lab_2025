## ️ Architektura Modeli i Metodologia

Przeprowadzono trening i ewaluację trzech modeli:

### 1. Baseline (Dense - Bez Augmentacji)
* **Architektura:** Prosta sieć w pełni połączona (Flatten $\rightarrow$ Dense 128 $\rightarrow$ Dense 10).
* **Trening:** Na oryginalnych danych treningowych.

### 2. Augmented Dense
* **Architektura:** Sieć w pełni połączona z dodaną warstwą augmentacji.
* **Trening:** Na danych treningowych z augmentacją.

### 3. CNN Augmented
* **Architektura:** Konwolucyjna Sieć Neuronowa (CNN) z warstwami Conv2D, MaxPooling2D oraz Dropout.
* **Trening:** Na danych treningowych z augmentacją.

###  Mechanizmy Augmentacji
Wprowadzona augmentacja na zbiorze treningowym obejmowała:
* `RandomTranslation` (przesunięcia o 10%).
* `RandomRotation` (rotacje o 20%).

Dodatkowo, **zbiór testowy do oceny odporności** został poddany tym samym transformacjom **oraz** losowemu odwróceniu kolorów (negatyw) z prawdopodobieństwem 30%, co stanowiło dużą trudność.

---

##  Wyniki Ewaluacji

Poniższa tabela prezentuje kluczowe wskaźniki dokładności (Accuracy) dla każdego modelu.

| Model | Acc. na Oryginalnym Z. Testowym | **Acc. na Augmentowanym Z. Testowym (Odporność)** | Różnica Acc. (Spadek) |
| :--- | :--- | :--- |:----------------------|
| **Baseline (Brak Aug.)** | $0.9753$ | **$0.3882$** | $58.71$ %             |
| **Augmented Dense** | $0.9368$ | **$0.8945$** | $4.23$ %              |
| **CNN Augmented** | $0.9814$ | **$0.9732$** | $0.82$ %              |

---

###  Wnioski

## 1. Konieczność Augmentacji Danych
* Model **Baseline** bez augmentacji, mimo wysokiej dokładności na czystych danych ($\approx 97.53\%$), wykazał katastrofalny spadek wydajności do **$38.82\%$** na zaugmentowanym zbiorze testowym. Oznacza to, że jest skrajnie **nieodporny** na proste zmiany (zwłaszcza na negatyw obrazu), co dyskwalifikuje go w realnych zastosowaniach.
* Wprowadzenie augmentacji do modelu Dense (**Augmented Dense**) podniosło odporność aż do **$89.45\%$**..

## 2. Przewaga Architektury CNN
* Model **CNN Augmented** osiągnął najlepsze wyniki w obu kategoriach: najwyższą dokładność na czystych danych ($\mathbf{98.14\%}$) i najlepszą odporność na dane przetworzone ($\mathbf{97.32\%}$).
## Podsumowanie Ostateczne
Połączenie **Architektury Konwolucyjnej (CNN)** z **Augmentacją Danych** jest optymalną strategią dla zadań widzenia komputerowego. Takie podejście nie tylko maksymalizuje dokładność, ale przede wszystkim zapewnia modelowi niezbędną **odporność i zdolność do uogólniania** na dane odbiegające od pierwotnego rozkładu.
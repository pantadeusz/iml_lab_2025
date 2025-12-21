# Raport z eksperymentów: Wine Dataset

**2. Random Forest (Scikit-Learn)**
Model osiągnął **100.00% dokładności**, co potwierdza, że zbiór jest łatwy do klasyfikacji dla algorytmów drzewiastych.
- **Rozmiar modelu:** 212.08 KB (stosunkowo duży).

**3. Prosta sieć neuronowa (bez normalizacji)**
Model 3-warstwowy (funkcja straty: *sparse_categorical_crossentropy*) osiągnął jedynie **80.56%**.
- **Wniosek:** Brak skalowania cech o różnych zakresach uniemożliwił skuteczną naukę.

**4. Sieć neuronowa z normalizacją**
Dodanie warstwy `Normalization` natychmiast podniosło wynik do **94.44%**.
- **Wniosek:** Normalizacja jest krytycznym etapem preprocessingu dla sieci neuronowych.

**5. Miniaturyzacja i Regularyzacja**
Przetestowano 8 konfiguracji w celu redukcji rozmiaru przy zachowaniu skuteczności.
- **100% dokładności** uzyskał model `Heavy_Reg` dzięki silnej regularyzacji L2, która zapobiegła przeuczeniu na małym zbiorze.
- **Najmniejszy model >97%** to `Deep_Narrow` (zaledwie **158 parametrów**!), co dowodzi skuteczności architektury typu "bottleneck".

**6. Podsumowanie**
Random Forest jest rozwiązaniem najpewniejszym, ale zajmuje najwięcej pamięci. Sieci neuronowe wymagają normalizacji, ale pozwalają na drastyczną redukcję rozmiaru. Najlepszy balans osiągnięto przy użyciu silnej regularyzacji (`Heavy_Reg`), uzyskując 100% skuteczności przy modelu wielokrotnie mniejszym niż Random Forest.
| Model / Technika                     | Precision (kl.1) | Recall (kl.1) | F1-score (kl.1) | Accuracy | Co się dzieje                                                                           |
| ------------------------------------ | ---------------- | ------------- | --------------- | -------- | --------------------------------------------------------------------------------------- |
| **Base model**                       | 0.25             | 0.09          | 0.13            | 0.935    | Model ignoruje klasę 1 – bardzo słaby recall, bo danych tej klasy jest mało.            |
| **Ważenie klas (`class_weight`)**    | 0.18             | 0.55          | 0.27            | 0.835    | Model zaczyna „widzieć” klasę 1, bo błędy na niej są bardziej karane.                   |
| **Oversampling (manualny)**          | 0.18             | 0.64          | 0.28            | 0.82     | Klasa 1 powielona – recall bardzo dobry, ale precision trochę spada.                    |
| **SMOTE**                            | 0.19             | 0.55          | 0.29            | 0.85     | Najlepszy balans między precision i recall – dane syntetyczne poprawiają generalizację. |
| **Undersampling (redukcja klasy 0)** | 0.13             | 0.55          | 0.21            | 0.77     | Model lepiej łapie klasę 1, ale traci dużo informacji o klasie 0 → spada accuracy.      |


Base model – bardzo wysoka dokładność, ale kompletnie ignoruje klasę mniejszościową.
Ważenie klas – dobry, prosty kompromis.
Oversampling – zwiększa recall, ale pogarsza precision.
SMOTE – najlepszy wynik ogólny (najwyższe F1 dla klasy 1).
Undersampling – działa, ale często zbyt agresywny — model traci ogólną dokładność, bo uczony jest na mało danych.
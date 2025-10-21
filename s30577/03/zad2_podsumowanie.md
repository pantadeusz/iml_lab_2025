# Zadanie 2 – Niezbalansowane klasy

## Metryki (accuracy / precision / recall / F1)
- Base: `0.940 / 0.250 / 0.062 / 0.100`
- Weighted: `0.853 / 0.196 / 0.562 / 0.290`
- Weighted (thr=0.3): `0.747 / 0.125 / 0.625 / 0.208`
- SMOTE: `0.860 / 0.205 / 0.562 / 0.300`
- Undersample: `0.830 / 0.182 / 0.625 / 0.282`

## Wnioski
- Bazowy model ignoruje klasę mniejszościową – wysoka dokładność, ale recall prawie zerowy.
- Ważenie klas i SMOTE znacząco poprawiają wykrywanie klasy 1. Oversampling SMOTE daje najwyższy F1 i pozostawia dobre accuracy.
- Obniżenie progu (0.3) zwiększa recall kosztem precyzji;
- Random undersampling skutecznie podnosi recall kosztem precyzji

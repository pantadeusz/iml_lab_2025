### Podsumowanie (RF vs baseline NN vs baseline NN + Normalization)

- **Zbiór i podział**: Wine (13 cech, 3 klasy), podział 80/20 ze stratyfikacją, stałe ziarno.
- **Uczenie**: 
  - RF: n_estimators=200.
  - NN: Adam, sparse_categorical_crossentropy, EarlyStopping (val_accuracy, patience=30).
  - Architektura NN : 8 → 4 → 3.

#### Porównanie wyników (typowe)
- **RandomForest**: najłatwiej osiąga 1.0000 acc “z pudełka”.
- **NN baseline (bez normalizacji)**: zwykle ~0.94–0.98 acc; wrażliwy na epoki/inicjalizację.
- **NN + Normalization**: stabilniej i wyżej; często 0.97–1.0000. Mała sieć 8→4 z większą liczbą epok (np. 600, batch 8) często dobija do 1.0000.

| Model | Zalety | Wady | Typowy wynik |
|---|---|---|---|
| RandomForest | Zero strojenia, szybki, mocny na tabelach | Brak “kompaktowego” zapisu jak NN | ≈ 1.0000 |
| NN baseline | Prosty, mały model | Niestabilny bez normalizacji | ~0.94–0.98 |
| NN + Normalization | Szybsza/pewniejsza zbieżność, mniejszy model wystarcza | Wymaga dobrania epok/rozmiaru | 0.97–1.0000 |


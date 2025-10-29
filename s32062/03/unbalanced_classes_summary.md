# Podsumowanie

## Wyniki modelu ze standardowym progiem

| Dane / strategia      | Precyzja | Czułość | F1-score | Dokładność |
|:----------------------|:--------:|:-------:|:--:|:----------:|
| Niezbalansowane klasy |  0.250   |  0.091  | 0.133 |   0.935    |
| Z ważeniem klas       |  0.176   |  0.545  | 0.267 |   0.835    |
| SMOTE                 |  0.194   |  0.545  | 0.286 |   0.850    |
| Losowe podpróbkowanie |  0.128   |  0.545  | 0.207 |   0.770    |

Po zastosowaniu technik balansowania klas, model bardzo zyskał na czułości i trochę na F1-score, ale nieco stracił na precyzji i dokładności.

## Wyniki modelu po modyfikacji progu decyzji

Czym niższy próg decyzji, tym model dalej zyskuje na na czułości, ale traci na wszystkich pozostałych metrykach.

Dla progu równego 0.1:

| Dane / strategia      | Precyzja | Czułość | F1-score | Dokładność |
|:----------------------|:--------:|:-------:|:--------:|:----------:|
| Niezbalansowane klasy |  0.250   |  0.091  |  0.133   |   0.935    |
| Z ważeniem klas       |  0.080   |  0.818  |  0.146   |   0.475    |
| SMOTE                 |  0.091   |  0.727  |  0.163   |    0.59    |
| Losowe podpróbkowanie |  0.075   |  0.818  |  0.138   |    0.44    |
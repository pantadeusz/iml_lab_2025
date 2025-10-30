# Zadanie 1 – Imputacja braków danych


## Wyniki (R2 / RMSE)
- Pełne dane: `0.477 / 53.12`
- 10% braków: `Mean 0.431 / 55.42`, `KNN 0.438 / 55.06`, `MICE 0.468 / 53.59`
- 20% braków: `Mean 0.460 / 53.99`, `KNN 0.473 / 53.33`, `MICE 0.486 / 52.66`

## Wnioski
- Nawet przy 10% braków widać lekki spadek jakości, szczególnie przy uśrednianiu – prosta imputacja pogarsza dopasowanie modelu.
- Metody wielowymiarowe (`KNN`, zwłaszcza `MICE`) utrzymują lub delikatnie poprawiają wyniki. Przy 20% braków MICE zbliżyło się do pełnych danych.
- W praktyce warto stosować bardziej zaawansowaną imputację niż sama średnia, a przy większym odsetku braków szczególnie opłaca się MICE.

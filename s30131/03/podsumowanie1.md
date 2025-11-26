## Cel
- Sprawdziłem, jak różne imputacje braków (MCAR/MAR/MNAR) wpływają na jakość modelu regresyjnego.

## Ustawienia
- Dane: `load_diabetes`; podział 80/20 (`random_state=42`).
- Pipeline: imputer → `StandardScaler` → `Ridge(alpha=1.0)`.
- Poziomy braków: 5% i 20%.

## Metody imputacji
- Średnia (Mean), KNN (k=5), MICE (IterativeImputer).

## Metryki
- R2 (↑ lepiej), MAE (↓ lepiej), RMSE (↓ lepiej).

## Najkrótsze wnioski
- Przy 5% braków Mean bywa ok, ale KNN/MICE zwykle wypadają lepiej.
- Przy 20% braków przewagę najczęściej ma KNN lub MICE; Mean traci najwięcej.
- MNAR jest najtrudniejszy (największy spadek jakości); MCAR zazwyczaj najłagodniejszy.
- Imputacja odzyskuje część jakości względem danych z brakami; brak braków (baseline) jest punktem odniesienia.

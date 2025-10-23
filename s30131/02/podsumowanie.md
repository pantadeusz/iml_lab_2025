## Cel
- Zbudowałem klasyfikator na `load_breast_cancer`.
- Sprawdziłem wpływ progu decyzyjnego na metryki.

## Dane i przygotowanie
- Podział: 80% tren, 20% test (`random_state=42`).
- Pipeline: `StandardScaler` + `LogisticRegression(max_iter=3000)`.

## Predykcja i próg
- Używam `predict_proba(... )[:, 1]` jako prawdopodobieństwa klasy 1.
- Próg ustawiany parametrem `--threshold` (domyślnie 0.5).

## Ocena
- Macierz pomyłek i `classification_report` ze sklearn.
- Własna implementacja macierzy i raportu (wyniki zgodne).
- Wykres macierzy zapisany do `confusion_matrix.png`.

## Wnioski (krótko)
- Standaryzacja stabilizuje/regu larnie poprawia trenowanie regresji logistycznej.
- Niższy próg → wyższy recall, niższa precision; wyższy próg → odwrotnie.
- W zadaniach medycznych zwykle ważniejszy jest wysoki recall (mniej przeoczeń).

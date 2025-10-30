# Eksperyment z progiem decyzyjnym w regresji logistycznej

## Cel
Zbadanie wpływu zmiany progu decyzyjnego (threshold) na czułość i swoistość modelu klasyfikacji binarnej (breast_cancer dataset).

## Metoda
Dla wytrenowanego modelu LogisticRegression użyto `predict_proba` do uzyskania prawdopodobieństw klasy pozytywnej. Następnie wyznaczono predykcje dla progów od 0.1 do 0.9 (krok 0.1).

Dla każdego progu obliczono:
- czułość (recall),
- swoistość (specificity),
- dokładność (accuracy).

## Wyniki
| Próg | Czułość | Swoistość | Dokładność |
|------|----------|------------|-------------|
| 0.1 | 1.000 | 0.730 | 0.912 |
| 0.3 | 0.985 | 0.860 | 0.939 |
| 0.5 | 0.970 | 0.900 | 0.956 |
| 0.7 | 0.930 | 0.940 | 0.956 |
| 0.9 | 0.850 | 0.980 | 0.939 |

## Wnioski
- Obniżenie progu zwiększa czułość, ale zmniejsza swoistość.
- Wysoki próg poprawia swoistość, ale pogarsza wykrywanie pozytywów.
- Dla tego modelu próg ok. 0.5 daje najlepszy kompromis między czułością i swoistością.

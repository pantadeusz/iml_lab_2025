## Cel
- Porównałem proste techniki na niezbalansowane klasy i ich wpływ na metryki.

## Dane i konfiguracja
- `make_classification` z niezbalansem ~95% vs 5%.
- Podział: train/val/test (60/20/20), stratyfikowany.
- Model bazowy: `LogisticRegression` + `StandardScaler`.
- Próg decyzyjny stroiłem na walidacji pod F1, testowałem na teście.

## Strategie
- Baseline (bez zabiegów).
- `class_weight='balanced'` w LogisticRegression.
- Oversampling: `SMOTE`.
- Undersampling: `RandomUnderSampler`.

## Metryki
- Accuracy, Precision, Recall, F1, ROC AUC, Average Precision (AUPRC).
- Dodatkowo pokazałem bazę przy stałym progu 0.5 jako punkt odniesienia.

## Najważniejsze wnioski (skrót)
- Samo strojenie progu zwykle mocno poprawia F1/recall wobec progu 0.5.
- `class_weight='balanced'` podnosi recall (często kosztem precision).
- `SMOTE` często daje najlepszy kompromis precision–recall i wyższy AUPRC przy dużym niezbalansowaniu.
- Undersampling bywa skuteczny dla recall, ale może obniżyć precision i stabilność (mniej danych).
- Dobór progu po walidacji jest kluczowy; bez tego wyniki bywają mylące przy klasach rzadkich.
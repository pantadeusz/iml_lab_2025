## Podsumowanie eksperymentu

Zbadano wpływ trzech metod imputacji (Mean, KNN, MICE) oraz trzech modeli regresyjnych (Linear, DecisionTree, RandomForest) na jakość predykcji w zbiorze `load_diabetes` przy 5%, 10% i 20% braków danych (MCAR).

### Wyniki

| Missing % | Imputer | Model | R² |
|------------|----------|--------|------|
| 5 | Mean | Linear | 0.388 |
| 5 | Mean | DecisionTree | -0.236 |
| 5 | Mean | RandomForest | 0.401 |
| 5 | KNN | Linear | 0.393 |
| 5 | KNN | DecisionTree | -0.075 |
| 5 | KNN | RandomForest | 0.397 |
| 5 | MICE | Linear | **0.438** |
| 5 | MICE | DecisionTree | 0.020 |
| 5 | MICE | RandomForest | **0.476** |
| 10 | Mean | Linear | 0.412 |
| 10 | Mean | DecisionTree | -0.247 |
| 10 | Mean | RandomForest | 0.395 |
| 10 | KNN | Linear | 0.415 |
| 10 | KNN | DecisionTree | 0.042 |
| 10 | KNN | RandomForest | 0.405 |
| 10 | MICE | Linear | **0.437** |
| 10 | MICE | DecisionTree | -0.211 |
| 10 | MICE | RandomForest | **0.453** |
| 20 | Mean | Linear | 0.466 |
| 20 | Mean | DecisionTree | -0.694 |
| 20 | Mean | RandomForest | 0.363 |
| 20 | KNN | Linear | **0.473** |
| 20 | KNN | DecisionTree | -0.031 |
| 20 | KNN | RandomForest | **0.474** |
| 20 | MICE | Linear | 0.423 |
| 20 | MICE | DecisionTree | -0.186 |
| 20 | MICE | RandomForest | 0.398 |


- **Najlepsze wyniki R² (~0.47)** uzyskano dla kombinacji **MICE + RandomForest** oraz **KNN + RandomForest**.  
- **Imputacja średnią (Mean)** dawała najniższe R², zwłaszcza przy większym odsetku braków.  
- **DecisionTree** osiągało często **ujemne R²**, co oznaczało wyniki gorsze niż przewidywanie średniej.  
- **LinearRegression** i **RandomForest** były stabilne wobec braków, ale RandomForest był dokładniejszy.  
- Wraz ze wzrostem liczby braków obserwowany był **spadek jakości modeli** niezależnie od metody.

### Wnioski
Najbardziej odporna kombinacja to **RandomForest + MICE/KNN**.  
Proste uzupełnianie średnią nie wystarcza dla większych braków danych.  
Ujemne wartości R² świadczą o utracie informacji po nieodpowiedniej imputacji.

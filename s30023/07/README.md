# Podsumowanie

- Dodanie warstwy normalizacyjnej bardzo polepsza wyniki i stabilizuje ich
- Dodanie Scaler pozwala zrobić model mniejszy i bardziej precyzyjny (zawsze wyniki 100%)

# Waga plików

- Random Forest: 215 KB
- Neural Network (3 layers): 68 KB

# Random Forest
```
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        14
           2       1.00      1.00      1.00        14
           3       1.00      1.00      1.00         8

    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36
```

# Bez warstwy normalizacji
```
                      precision    recall  f1-score   support

           0       0.48      1.00      0.65        14
           1       0.86      0.43      0.57        14
           2       0.00      0.00      0.00         8

    accuracy                           0.56        36
   macro avg       0.45      0.48      0.41        36
weighted avg       0.52      0.56      0.48        36
```
# Z warstwą normalizacji 
```
                      precision    recall  f1-score   support

           0       1.00      0.93      0.96        14
           1       0.93      1.00      0.97        14
           2       1.00      1.00      1.00         8

    accuracy                           0.97        36
   macro avg       0.98      0.98      0.98        36
weighted avg       0.97      0.97      0.97        36
```
# Z warstwą normalizacji + Scaler
```
                      precision    recall  f1-score   support

           0       1.00      1.00      1.00        14
           1       1.00      1.00      1.00        14
           2       1.00      1.00      1.00         8

    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36
```
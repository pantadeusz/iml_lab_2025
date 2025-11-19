# Podsumowanie z porównania dwóch metod uczenia: RandomForest oraz prostej DNN
### Zbiór testowy: load_brest_cancer() z scikit-learn, oraz random seed = 50

## Wyniki dla RandomForest:

|  | precision | recall | f1-score | support |
|:-----:|:---------:|:------:|:--------:|:-------:|
|0      |95%        | 95%    | 95%      | 42      |
|1      |97%        |97%     |97%       |72       |
| | | |
|accuracy|          |        | 96%      |114      |
|macro avg|96%|96%|96%|114|
|weighted avg|96%|96%|96%|114|

## Wyniki dla DNN

|  | precision | recall | f1-score | support |
|:-----:|:---------:|:------:|:--------:|:-------:|
|0      |100%        | 95%    | 97%      | 42      |
|1      |97%        |100%     |98%       |72       |
| | | |
|accuracy|          |        | 98%      |114      |
|macro avg|98%|97%|98%|114|
|weighted avg|98%|98%|98%|114|

## Wnioski:
Model DNN uzyskał nieco wyższą dokładność, f1-score oraz recall w porównaniu do RandomForest, co oznacza, że skuteczniej klasyfikował przypadki klasy pozytywnej.

Random Forest, mimo prostoty i braku potrzeby standaryzacji danych, nadal oferuje bardzo stabilne wyniki i mniejszą podatność na przeuczenie przy małych zbiorach danych. DNN z kolei pokazuje potencjał do uzyskania lepszych rezultatów, jeśli dane są odpowiednio przetworzone i model ma właściwą regularyzację.
=== Imputation: Mean || Missing 5.0% ===
MSE:  3173.825
RMSE: 56.337
MAE:  45.155
R²:   0.401

=== Imputation: KNN || Missing 5.0% ===
MSE:  3196.401
RMSE: 56.537
MAE:  45.250
R²:   0.397

=== Imputation: MICE || Missing 5.0% ===
MSE:  2801.843
RMSE: 52.932
MAE:  42.942
R²:   0.471

=== Imputation: Mean || Missing 10.0% ===
MSE:  3203.249
RMSE: 56.597
MAE:  44.156
R²:   0.395

=== Imputation: KNN || Missing 10.0% ===
MSE:  3153.143
RMSE: 56.153
MAE:  44.455
R²:   0.405

=== Imputation: MICE || Missing 10.0% ===
MSE:  2975.120
RMSE: 54.545
MAE:  43.237
R²:   0.438

=== Imputation: Mean || Missing 50.0% ===
MSE:  4421.011
RMSE: 66.491
MAE:  52.265
R²:   0.166

=== Imputation: KNN || Missing 50.0% ===
MSE:  4163.994
RMSE: 64.529
MAE:  53.135
R²:   0.214

=== Imputation: MICE || Missing 50.0% ===
MSE:  4359.895
RMSE: 66.030
MAE:  53.585
R²:   0.177

### Wniąski:

Wraz ze wzrostem odsetka brakujących danych, skuteczność predykcyjna wszystkich metod imputacji konsekwentnie spada.
Spośród nich imputacja oparta na modelu (MICE) osiąga najlepszą dokładność w przypadku niskiego lub umiarkowanego poziomu braków, 
natomiast imputacja KNN wykazuje większą odporność w przypadku ekstremalnej utraty danych.

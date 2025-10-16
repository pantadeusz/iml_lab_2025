Stan bazowy
    Regresja logistyczna wydaje się być najlepiej sprawdzającym się algorytmem, ma największe pole pod krzywą ROC or PR.

Eksperymenty
    Regresja logistyczna
    Już przy 50 iteracjach osiąga lepsze wyniki niż bazowy RandomForest i StandardVectorMachine. Lepszych wyników nie udało się osiągnąć, stosowane były zmiany parametrów: C, penalty, solver, max_iter.

    RandomForest (BEST)
    Używane były parametry n_estimators, max_depth, min_samples_split, criterion, class_weight. Udało się uzyskać lepszy wynik niż bazowa regresja logistyczna przy jednej zmianie funkcji kosztu - criterion='log_loss'.

    SVM
    Nie udało się polepszyć wyników algorytmu strojąc parametry: C, kernel, gamma, degree.

Niezbalansowany dataset
    Krzywa PR ma znacznie mniejsze AUC. SVM wydaje się być lepszy przy takiego rodzaju datasetach. RandomForest wydaje się być najlepszy ponownie. Zmiany jego parametrów nie wprowadzały znacznych różnic.
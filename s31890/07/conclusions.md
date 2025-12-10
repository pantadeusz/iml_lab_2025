## Infrastruktura
#### Dane
* Modele trenowane na zbiorze danych [Wina](https://archive.ics.uci.edu/dataset/109/wine)

#### Przeprowadziłem 90 eksperymentów używając keras tuner, oto 5 najlepszych modeli.

|Trial ID|Score             |Hyperparameters                                                                                                                                                                                                                                                                                                |
|--------|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|0006    |1.0               |{'activation': 'relu', 'use_batch_norm': False, 'num_dense_layers': 2, 'units_per_dense': 192, 'dropout_rate': 0.25, 'learning_rate': 0.004299211781488068, 'optimizer': 'adam', 'tuner/epochs': 2, 'tuner/initial_epoch': 0, 'tuner/bracket': 3, 'tuner/round': 0}                                            |
|0018    |1.0               |{'activation': 'elu', 'use_batch_norm': False, 'num_dense_layers': 2, 'units_per_dense': 288, 'dropout_rate': 0.2, 'learning_rate': 0.0003047566454213899, 'optimizer': 'rmsprop', 'tuner/epochs': 2, 'tuner/initial_epoch': 0, 'tuner/bracket': 3, 'tuner/round': 0}                                          |
|0021    |1.0               |{'activation': 'elu', 'use_batch_norm': True, 'num_dense_layers': 4, 'units_per_dense': 448, 'dropout_rate': 0.0, 'learning_rate': 0.0027492898886607555, 'optimizer': 'adam', 'tuner/epochs': 2, 'tuner/initial_epoch': 0, 'tuner/bracket': 3, 'tuner/round': 0}                                              |
|0022    |1.0               |{'activation': 'relu', 'use_batch_norm': False, 'num_dense_layers': 1, 'units_per_dense': 288, 'dropout_rate': 0.25, 'learning_rate': 0.009458590727480821, 'optimizer': 'rmsprop', 'tuner/epochs': 2, 'tuner/initial_epoch': 0, 'tuner/bracket': 3, 'tuner/round': 0}                                         |
|0025    |1.0               |{'activation': 'elu', 'use_batch_norm': True, 'num_dense_layers': 2, 'units_per_dense': 384, 'dropout_rate': 0.0, 'learning_rate': 0.00019071203207567507, 'optimizer': 'rmsprop', 'tuner/epochs': 2, 'tuner/initial_epoch': 0, 'tuner/bracket': 3, 'tuner/round': 0}                                          |

## Wniąski
#### Największa dokładność, a architektura modeli
* Bardzo wiele konfiguracji osiągnęło maksymalną dokładność, dla wyników wszystkich prób proszę zobaczyć "tuner_trials_summary.csv".
* 100% dokładności jesteśmy wstanie osiągnąć każdą liczbą warstw 'dense' branych pod uwagę w eksperymencie (od 1 do 4).
* Ciężko zbadać skuteczność warstw normalizacyjnych 'batch_norm', ponieważ z nimi jak i bez nich model jest wstanie osiągnąć 100% poprawności.

* Random forest classifier równierz osiąga 100% dokładności. Jest to zgodne z wynikami podanymi na stronie internetowej zbioru danych.
 
#### Wykresy
* skrypt plot.py generuje trzy interesujące wykresy na podstawie 'tuner_trials_summary.csv'
* W szczególności polecam zobaczenie wykresu 'scatter' score vs learning rate.

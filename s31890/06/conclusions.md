## Infrastruktura

#### Dane 
* Modele trenowane na zbiorze danych fasoli 'beans'.
* Postanowiłem użyć obrazków w pełnej rozdzielczości (500x500p), co uzasadnione było wieloma podobieństwami pomiędzy obrazami liści z różnymi etykietami. Z perspetywy czasu uważam to za błąd, ponieważ koszty obliczeniowe które to przyniosło były ogromne.


#### Przeprowadziłem 132 eksperymenty używając keras tuner, oto 5 najlepszych modeli.

|Trial ID|Score              |Hyperparameters                                                                                                                                                                                                                                                                                                                                                                                        |
|--------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|0065    |0.8270676732063293 |{'num_conv_blocks': 3, 'activation': 'relu', 'use_batch_norm': True, 'num_dense_layers': 2, 'units_per_dense': 448, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.0009513380195031257, 'optimizer': 'adam', 'conv_filters_0': 64, 'conv_filters_1': 128, 'conv_filters_2': 128, 'tuner/epochs': 30, 'tuner/initial_epoch': 0, 'tuner/bracket': 0, 'tuner/round': 0}                          |
|0128    |0.8270676732063293 |{'num_conv_blocks': 3, 'activation': 'relu', 'use_batch_norm': True, 'num_dense_layers': 4, 'units_per_dense': 448, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.009131857765102563, 'optimizer': 'sgd', 'conv_filters_0': 32, 'conv_filters_1': 32, 'conv_filters_2': 64, 'tuner/epochs': 30, 'tuner/initial_epoch': 0, 'tuner/bracket': 0, 'tuner/round': 0}                              |
|0130    |0.8195488452911377 |{'num_conv_blocks': 2, 'activation': 'elu', 'use_batch_norm': True, 'num_dense_layers': 3, 'units_per_dense': 128, 'dropout_rate': 0.5, 'learning_rate': 0.0010848539808730223, 'optimizer': 'adam', 'conv_filters_0': 32, 'conv_filters_1': 32, 'conv_filters_2': 64, 'tuner/epochs': 30, 'tuner/initial_epoch': 0, 'tuner/bracket': 0, 'tuner/round': 0}                                             |
|0127    |0.8120300769805908 |{'num_conv_blocks': 3, 'activation': 'elu', 'use_batch_norm': False, 'num_dense_layers': 4, 'units_per_dense': 448, 'dropout_rate': 0.2, 'learning_rate': 0.0003998392124307353, 'optimizer': 'adam', 'conv_filters_0': 32, 'conv_filters_1': 32, 'conv_filters_2': 128, 'tuner/epochs': 30, 'tuner/initial_epoch': 0, 'tuner/bracket': 0, 'tuner/round': 0}                                           |
|0062    |0.8120300769805908 |{'num_conv_blocks': 3, 'activation': 'elu', 'use_batch_norm': True, 'num_dense_layers': 2, 'units_per_dense': 448, 'dropout_rate': 0.30000000000000004, 'learning_rate': 0.00012740002244866934, 'optimizer': 'rmsprop', 'conv_filters_0': 32, 'conv_filters_1': 32, 'conv_filters_2': 64, 'tuner/epochs': 10, 'tuner/initial_epoch': 0, 'tuner/bracket': 1, 'tuner/round': 0}                         |



## Wniąski
#### Największa dokładność, a architektura modeli
* Dwa najlepsze modele osiągnęły identyczną dokładność, pomimo różnic w architekturze.
  - Jeden z nich używając optymalizatora 'adam' z około 0.00095 learning rate, dwoma warstwami 'dense' oraz 3 blokami konwolucyjnymi
  - Drugi z nich używał optymalizatora 'sgd' ze znacznie większym learning rate na poziomie w okolicach 0.009, czteroma warstwami 'dense' oraz tą samą ilością bloków konwolucyjnych. Bloki konwolucyjny miały zupełnie inne wartości, znacznie niższe niż w przypadku modelu z 'adam'.

* Udowadnia to, że możliwe jest osiągnięcie podobnej a wręcz w tym przypadku tej samej maksymalnej dokładności za pomocą różnych optymalizatorów, kompensując różnymi prędkościami nauki jak architekturą.

#### Znaczenie kateogrycznych hiperparametrów
* Używanie normalizacji partii 'batch norm' pozwoliło uzyskać największą dokładność czterem z pięciu najlepszych modeli, lecz jest możliwe uzyskanie wysokiej dokładności (miejsce 4) bez niej.

* Co ciekawe ilość wartsw gęstych 'dense' jest bardzo różna w zakresie od 2 do 4 wśród pięciu najlepszych modeli.

#### Znaczenie hiperparametrów numerycznych
* Liczba jednostek 'units' w czterech z pięciu najlepszych modeli jest wysoka (448).
* 'drouput rate' o wartości w okolicach 0.3 ma pozytywny wpływ na dokładność modelu.
 
#### Wykresy
* skrypt plot.py generuje dwa interesujące wykresy na podstawie 'tuner_trials_summary.csv'
* W szczególności polecam zobaczenie wykresu 'scatter' score vs learning rate.

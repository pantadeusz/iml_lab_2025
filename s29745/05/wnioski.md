## Wnioski do zadania z Labu 5

### Cel zadania
Celem zadania było stworzenie dwóch modeli - jeden z biblioteki scikit-learn, a drugi miał być siecią DNN z autotunerem opertą na TensorFlow. Podrzędnym celem było sprawienie, aby model DNN osiągnął lepsze wyniki niż model z scikit-learn.

### Opis architektury
Za bazowy model posłużył mi **RandomForest**, natomiast mój model DNN składał się z następujących warstw:
- **Warstwa wejściowa** z liczbą neuronów równej liczbie danych po preprocessingu
- **Warstwa ukryta 1** ze zmienną liczbą neuronów (32-512), funkcją aktywacji ReLu oraz ze zmiennym dropout z zakresu 0.1-0.5
- **Warstwa ukryta 2** ze zmienną liczbą neuronów (16-256), funkcją aktywacji ReLu oraz ze zmiennym dropout z zakresu 0.1-0.5 
- **Warstwa wyjściowa** 1 neuron z funkcją aktywacji Sigmoid

### Opis eksperymentu
Eksperyment miał na celu porównanie klasycznego RandomForest z głęboką siecią neuronową wzbogaconą o autotuner w zadaniu klasyfikacji binarnej dochodu. Za dataset posłużył mi zbiór **Adult** pobierany z **openml**.
<br><br>
Ze względu na charakterystykę wybranego przeze mnie zbioru dane musiały być mocno przetworzone:
- imputacja brakujących wartości (**IterativeImputer** dla danych numerycznych oraz **SimpleImputer** dla kategorii)
- normalizacja danych numerycznych za pomocą **ScandardScaler**
- kodowanie kategorii za pomocą **OneHotEncoder**
- dane *target* zostały zmapowane (0 - osoby zarabiające < 50k, 1 - osoby zarabiające >= 50k)

### Parametry autotunera
- Typ tunera: **RandomSearch**
- Metryka celu: **val_accuracy**
- Liczba prób (**max trials**): 20
- Liczba trenowań na próbę: 1
- Liczba epok treningu: 20
- Callback: **EarlyStopping**
<br>***Zmienne hiperparametry***:
- **units_1**: 32-512, step 32
- **dropout_1**: 0.1-0.5, step 0.1
- **units_2**: 16-256, step 16
- **dropout_2**: 0.1-0.5, step 0.1
- learning_rate: [0.001, 0.0005, 0.0001]

### Wyniki RandomForest

| Klasa            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.8819    | 0.9294 | 0.9050   | 7431    |
| 1                | 0.7291    | 0.6044 | 0.6609   | 2338    |
| **Accuracy**     | -         | -      | 0.8516   | 9769    |
| **Macro avg**    | 0.8055    | 0.7669 | 0.7829   | 9769    |
| **Weighted avg** | 0.8453    | 0.8516 | 0.8466   | 9769    |


### Wyniki DNN z autotunerem

| Klasa            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.8878    | 0.9253 | 0.9062   | 7431    |
| 1                | 0.7258    | 0.6283 | 0.6735   | 2338    |
| **Accuracy**     | -         | -      | 0.8542   | 9769    |
| **Macro avg**    | 0.8068    | 0.7768 | 0.7899   | 9769    |
| **Weighted avg** | 0.8490    | 0.8542 | 0.8505   | 9769    |

### Wnioski
Oba modele osiągnęły podobne wyniki. Nie udało mi się sprawić, aby DNN wykazał przewagę nad RanfomForest. Prawdopodobnie wynika to z charakterystyki datasetu, ponieważ w gruncie rzeczy jest to dataset tabelaryczny, w których RandomForest radzi sobie dobrze. Eksperymentowanie z parametrami autotunera przynosiły zmiany na poziomie na tyle znikomym, że nie jest to warte notowania.

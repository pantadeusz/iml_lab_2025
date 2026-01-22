## Wprowadzenie
Zadaniem jest opracowanie modelu, który przewiduje wartości akcji w kolejnych n dniach.

Program ma możliwość podania wybranej akcji w argumencie np. AAPL czy GOOG. Dane pobierane są z yahoo finance przy pomocy biblioteki `yfinance`. Dokonywana jest predykcja następnych n wartości również z możliwością wprowadzenia za pomocą argumentu.

Przykładowe użycie:
```bash
python main.py --name INTC --steps 20
```

W tym przypadku zostaną przewidziane wartości akcji Intel Corporation w 20 następnych dniach.

## Preprocessing
Zastosowana została normalizacja danych `dane = (dane - mean) / std`. Dodatkowo zestaw danych został podzielony na zbiory: treningowy (70%), walidacyjny (20%) oraz testowy (10%).
Kolumna `Volume` został przekonwertowany z użyciem `np.log1p`.
Próbowałem ustawić kolumnę cykliczną na dzień oraz rok, jednak coś mi nie wychodziło, więc zrobiłem predykcję bez nich.

> Wykres kolumny Close
![](https://i.imgur.com/Iqn1NZE.png)

> Wykres kolumny Volume po transformacji logarytmicznej
![](https://i.imgur.com/EVFDh1p.png)


## Modele
### Baselines
1. Powtórzenie wartości w ostatnim punkcie w czasie
2. Powtórzenie ponownie wartości z przedziału czasowego

### Liniowy
* **Linear**
    * **Units**: liczba kroków do przewidzenia * liczba cech

### Nieliniowy gęsty
* **Linear**
    * **Units**: 512 + ReLU
* **Linear**
    * **Units**: liczba kroków do przewidzenia * liczba cech

### Konwolucyjny
* **Conv1D**:
    * **Units**: 256 + ReLU
* **Linear**
    * **Units**: liczba kroków do przewidzenia * liczba cech

### Rekurencyjny
* **LSTM**
    * **Units**: 32
* **Linear**
    * **Units**: liczba kroków do przewidzenia * liczba cech

Wszystkie architektury zawierają końcową warstwę zmieniającą kształt wyjścia: `Reshape([steps, num_features])`

## Trening
Trening każdego modelu zostały ustawiony na 20 epok. Funkcja kosztu - MSE, optymalizator Adam z współczynnikiem uczenia na 0.001. Zastosowane zostało również wczesne zatrzymanie z parametrem cierpliwości równym 2. Dane wykorzystywane do treningu - wartości akcji sprzed ostatnich 10 lat z odstępem jednego dnia.

## Wyniki
Wyniki inferencji poszczególnych modeli.
* **Baseline (powtórzenie ostatniej wartości)**
![](https://i.imgur.com/3CDiiUf.png)

* **Baseline (powtórzenie całej sekwencji)**
![](https://i.imgur.com/x2z46hh.png)

* **Liniowy**
![](https://i.imgur.com/YfTn4gN.png)

* **Nieliniowy gęsty**
![](https://i.imgur.com/pH7tnzU.png)

* **Konwolucyjny**
![](https://i.imgur.com/6uKp7wm.png)

* **Rekurencyjny**
![](https://i.imgur.com/F9byqhE.png)

Modele porównywane są na podstawie metryki MAE.
![](https://i.imgur.com/mBHdPzN.png)

## Wnioski
Do zadania polegającego na predykcji następnych wartości cen akcji zostało użytych 6 architektur, 2 bazowe powtarzające poprzednie wartości, liniowa, nieliniowa, konwolucyjna oraz rekurencyjna. 

Trening przeprowadzany był na 20 epokach z early stoppingiem z patience=2. Został użyty Adam jako optymalizator oraz MSE jako funkcja kosztu. Dodatkowo została użyta metryka MAE do ewaluacji modeli.

Najlepsze okazały się być te baselinowe. Bardziej skomplikowane architektury wypadały całkiem nieźle na zbiorze walidacyjnym jednak okazywały się być tragiczne w przypadku predykcji na zbiorze testowym. 

Z jakiego powodu? Bardzo możliwe, że predykcja cen akcji nie jest zadaniem tak bardzo cyklicznym jak prognoza pogody, pory roku, temperatura tylko bardziej polega na wydarzeniach ze świata, bądź cechach, które nie zostały umieszczone w datasecie.
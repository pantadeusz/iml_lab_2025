# Lab 11 - Predykcja następnej wartości w szeregu czasowym


Dzisiaj znowu bazujemy na [samouczku TensorFlow o szeregach czasowych](https://www.tensorflow.org/tutorials/structured_data/time_series)

Zadanie na dzisiaj:

Postaraj się przewidzieć następny kurs dzinny albo miesięczny.

Na stronie Yahoo można pobrać wykresy kursów akcji. Na przykład 

[Yahoo Finance - AAPL History (Weekly)](https://finance.yahoo.com/quote/AAPL/history/?frequency=1wk)

Jeśli chcemy mieć dane do przetwarzania, można zobaczyć jakie odpowiedzi zwraca API Yahoo. Na przykład

[Yahoo Finance API - AAPL Chart Data](https://query1.finance.yahoo.com/v8/finance/chart/AAPL?events=capitalGain%7Cdiv%7Csplit&formatted=true&includeAdjustedClose=true&interval=1wk&period1=1733893965&period2=1765429965&symbol=AAPL&userYfid=true&lang=en-US&region=US)

## Zadanie na dziś

Przygotuj proste konsolowe narzędzie do predykcji ceny akcji w następnym kroku czasowym.

Dane uczące powinny być w formacie JSON - prosto ze strony

Powinien tym razem powstać jeden program który będzie uczył RNN na danych giełdowych. Na koniec powinien pokazać kolejną predykcję i średni błąd kwadratowy. Proszę pamiętać o podziale na zbiory treningowy i testowy.

Przetrenuj na zdalnej maszynie udostępnionej do zajęć.

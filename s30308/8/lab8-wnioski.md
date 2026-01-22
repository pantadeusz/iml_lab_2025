# Eksperement z augmentacją danych #

## Początkowy model przed wykonaniem zadania ##

- Liczba epok = 10
- Rozmiar batch'a = 128
- Algorytm uczenia = Adam

## Model baseline przed augmentacją ##

- Loss: 0.073
- Accuracy: 0.979

## Augmentacja dla zbioru testowego ## 

- Loss: 16.676
- Accuracy: 0.454

Widać spadek jakości modelu. Model przestał rozpoznawać zdjęcie przez jasne tło oraz lekkie obroty.
Po prostu model uczył się rozkładu pikseli na obrazie zamiast faktycznych kształtów.

## Augmentacja zbioru treningowego i testowego ##

- Loss: 0.184
- Accuracy: 0.943

Spadła dokładność w porównaniu do baseline jednakże ten model lepiej generalizuje

## Model konwolucyjny z augmentacją danych ##

- Loss: 0.176, 
- Accuracy: 0.946

## Wnioski z zadania

- Model baseline osiągał najlepszą dokładność ale kosztem generalizacji
- Model baseline z augmentacją dobrze generalizuje i ma wysoką dokładność
- Model konwoluncyjny jest nieznaczne lepszy od sieci FNN z augmentacją
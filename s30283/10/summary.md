## Wprowadzenie
Zadanie polega na analizie sentymentu tekstu z użyciem warstw rekurencyjnych w sieciach neuronowych.

## Zbiór danych
Dataset wykorzystany do eksperymetów - IMDB large movie review dataset. Jest to zbiór danych wykorzystywany do klasyfikacji binarnej. Polega na analizie sentymentu opinii użytkowników o filmach - pozytywna lub negatywna.

## Architektura modelu
| Wartstwa | Output Shape | Liczba Parametrów |
| :--- | :--- | :--- |
| Text Vectorization | (None, None) | 0 |
| Embedding | (None, None, 64) | 64 000 |
| Bidiretional LSTM | (None, 128) | 66 048 |
| Dense | (None, 64)  | 8 256 |
| Dense | (None, 1) | 65 |

Total params: 415,109 (1.58 MB)
Trainable params: 138,369 (540.50 KB)
Non-trainable params: 0 (0.00 B)
Optimizer params: 276,740 (1.06 MB)

## Trening
Model trenowany był na parę sposobów. Powiększyłem również liczbę epok i zmniejszyłem rozmiar batcha byśmy mogli dokładniej przypatrzeć się zmianie dokładności i stracie na przestrzeni czasu.

Początkowo w preprocessingu input jest przerabiany na sekwencje tokenów za pomocą warstwy TextVectorization.
Następnie przechodzi przez warstwę Embedding, która generuje embeddings dla każdego tokenu wymiaru 64.
Potem embeddingi przechodzą przez warstwę rekurencyjną czyli dwustronny LSTM, gdzie model uczy się zależności i kontekstu.
Na końcu znajduje się warstwa w pełni połączona, która tworzy 64 cechy, dzięki którym może dostrzegać różne relacje pomiędzy tokenami.

* Optymalizator - Adam, learning rate = 0.001
* Funkcja straty - binary crossentropy

### Przebiegi treningów dla poszczególnych ustawień
> `batch_size` = 64, `epochs` = 10
![](https://i.imgur.com/IxY7GIl.png)

> `batch_size` = 64, `epochs` = 20
![](https://i.imgur.com/GakyaLQ.png)

> `batch_size` = 64, `epochs` = 30
![](https://i.imgur.com/KeZQT56.png)

> `batch_size` = 32, `epochs` = 20
![](https://i.imgur.com/P2ElFZ5.png)

> `batch_size` = 128, `epochs` = 20
![](https://i.imgur.com/GKBAiZl.png)


## Dodatkowe funkcje
Stworzony został również plik `runner.py`, odpowiedzialny za przetestowanie działania wytrenowanego modelu. Wystarczy posłużyć się poniższą instrukcją, aby dokonać analizy sentymentu tekstu, który podamy w inpucie w konsoli:

```sh
python runner.py --path (ścieżka do modelu)
> (Tutaj należy wprowadzić text)
Następnie wyświetli się odpowiedź modelu - (pozytywna | negatywna) oraz prawdopodobieństwo predykcji
```

## Wnioski
Problem polegał na wytrenowaniu modelu do efektywnej analizy sentymentu tekstów. 

Trenowany był na IMDB reviews w różnych kombinacjach rozmiaru batcha oraz liczby epok. 

Można dostrzec, że model przeuczał się po mniej więcej 10 epokach, po zaburzeniu stabilności metryk walidacyjnych. Najmniejszy rozmiar batcha  (32) okazał się najmniej stabilny. Generalnie modele wykazywały bardzo podobne wyniki. Szczyty osiągały w okolicach 0.87 dokładności walidacyjnej, jednak po 10 epokach dokładność miała tendencję do niewielkiej regresji i redukcji stabilności.

Dodatkowo został zaimplementowany program do testowania wytrenowanego modelu na żywo, który bierze input z konsoli i mówi nam czy nasza opinia jest pozytywna, czy negatywna. 
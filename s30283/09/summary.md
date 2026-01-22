### Wprowadzenie
Zadaniem jest użycie Autoencodera do przekształcania obrazów po losowej niewielkiej rotacji do pierwotnego położenia.

### Zbiór danych
Data set wykorzystany do eksperymetów - Fashion MNIST. Posiada 60 000 zdjęć w rozmiarze 28x28 w gamie szarości przedstawiający ubrania.

### Testowane architektury

* **Bazowy:**
    > Brak dodatkowych warstw. Encoder po prostu transformuje do latent dima, natomiast Decoder dekoduje z niego informacje.
    * `latent_dim`: 64
* **Z warstwami konwolucyjnymi:**
    > Dodatkowe dwie warstwy konwolucyjne oraz pooling 2x2 dodane po każdej z nich w Encoderze.
    * **Encoder:**
        * `Conv2D` filtry(64), aktywacja(ReLU), kernel(3x3)
        * `MaxPooling2D` rozmiar(2x2)
        * `Conv2D` filtry(32), aktywacja(ReLU), kernel(3x3)
        * `MaxPooling2D` rozmiar(2x2)
    * `latent_dim`: 64 (tak jak w poprzednim przypadku)

### Trening
Każdy z modeli był trenowany przez **10 epok** na danych **w batchach o rozmiarze 32**. Optimalizator **Adam**, funkcja kosztu **MSE**.

### Wynik inferencji poszczególnych architektur
* **Bazowy:**
![Wyniki bazowego autoencodera](https://i.imgur.com/hdfB4dt.png)

* **Z konwolucjami w encoderze:**
![Wyniki autoencodera z konwolucjami](https://i.imgur.com/7NsxzWu.png)

### Podsumowanie
Przetestowane zostały dwie architektury Autoencodera do przywracania pierwotnego wyglądu zdjęć ze zbioru danych Fashion MNIST po losowych rotacjach. 

Z wyników inferencji można wyczytać, że bazowy oraz konwolucyjny model autoencodera naprawdę dobrze doprowadzały do pierwotnego stanu obrazów, jednak pozostawiały dosyć sporo szumu i nie zwracały szczególnej uwagi na detale rdzennych zdjęć. 

Ciekawą rzeczą również jest szum powstały w przypadku obrazka 3, który przypomina buta znajdującego się za ubraniem.

Nie dostrzeżona została znacząca różnica pomiędzy wynikami inferencji modelu bazowego oraz zawierającego warstwy konwolucyjne.

### Dodatkowe informacje
Stworzony został również plik `runner.py` odpowiedzialny za testowanie inferencji poszczególnych architektur na życzenie.

Plik przyjmuje argumenty wskazujące ścieżki do encodera, decodera oraz obrazka, na którym ma być wykonana inferencja. W dodatku wyświetlany jest wektor latent.

**Wyniki modeli:**
* **Bazowy:** ![Bazowy](https://i.imgur.com/cQ9CQcE.png)
* **Konwolucyjny:** ![Konwolucyjny](https://i.imgur.com/nPGPvMh.png)

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
Każdy z modeli był trenowany przez **10 epok** na danych **w batchach o rozmiarze 32**.

### Wyniki inferencji poszczególnych architektur
* **Bazowy:**
!(Wyniki bazowego autoencodera)[]

* **Z konwolucjami w encoderze:**
!(Wyniki autoencodera z konwolucjami)[]

### Podsumowanie
Przetestowane zostały dwie architektury Autoencodera do przywracania pierwotnego wyglądu zdjęć ze zbioru danych Fashion MNIST po losowych rotacjach. 
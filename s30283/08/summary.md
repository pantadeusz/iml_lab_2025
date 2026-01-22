### Zbiór danych
MNIST - rozpoznawanie liczb, 10 klas

### Uczenie modeli
Każdy model uczony był wedle podanych parametrów:
* **Epoki:** 10
* **Rozmiar batcha:** 32
* **Algorytm uczenia:** Adam
* **Współczynnik uczenia:** 0.001

### Modele
* **Bazowy**: 
    * Połączona - **128** unitów, ReLU
* **Konwolucyjny**
    * **Konwolucyjna** - **64** filtry, ReLU
    * **Pooling** - rozmiar **2x2**
    * **Konwolucyjna** - **32** filtry, ReLU
    * **Pooling** - rozmiar **2x2**
    * **Połączona** - **128** unitów, ReLU
    * **Połączona** - **32** unitów, ReLU

### Wyniki modelu
Wyniki przedstawiają dokładność (sparse categorical accuracy) osiągniętą przez poszczególne architektury.

| Model | Na czym trenowany | Na czym testowany | Dokładność |
| :--- | :--- | :--- | :--- |
| Bazowy | Pierwotny zbiór treningowy | Pierwotny zbiór walidacyjny | 0.98 |
| Bazowy | Pierwotny zbiór treningowy | Zbiór walidacyjny po transformacjach | 0.26 |
| Bazowy | Połączenie pierwotnego zbioru treningowego z tym po transformacjach | Pierwotny zbiór walidacyjny | 0.97 |
| Konwolucyjny | Pierwotny zbiór treningowy | Pierwotny zbiór walidacyjny | 0.99
| Konwolucyjny | Pierwotny zbiór treningowy | Zbiór walidacyjny po transformacjach | 0.45
| Konwolucyjny | Połączenie pierwotnego zbioru treningowego z tym po transformacjach | Pierwotny zbiór walidacyjny | 0.99

### Wnioski
Modele były trenowane i testowane na trzy sposoby:

| Na czym trenowany | Na czym testowany |
| :--- | :--- |
| Pierwotny zbiór treningowy | Pierwotny zbiór walidacyjny |
| Pierwotny zbiór treningowy | Zbiór walidacyjny po transformacjach |
| Połączenie pierwotnego zbioru treningowego z tym po transformacjach | Pierwotny zbiór walidacyjny |

Modelem bazowym (1 warstwa ukryta - 128 unitów) udało się osiągnąć 98% dokładności w pierwszym układzie, 22% (bardzo mało) w drugim oraz 97% w trzecim - ciekawy fakt, ponieważ model powinien wiedzieć więcej ze względu na dodatkowe dane przetransformowane, ale z drugiej strony jakość tych transformacji mogła również przyczynić się do niewystarczająco dokładnych danych co wpłynęło na naukę.

Model konwolucyjny (64 filtry > pooling 2x2 > 32 filtry > pooling 2x2) okazał się być lepszy w każdym przypadku.
W pierwszym procesie uzyskał 99%, tak samo ile w drugim 99%, jednak ta wartość była lekko mniejsza (wyniki dokładności podane w zaokrągleniu), co również wskazuje na podobny efekt zaobserwowany w przypadku eksperymentów na poprzednim modelu. Model ten również uzyskał dwa razy większą dokładność na przetransformowanych danych walidacyjnych - 45%.

### Przykładowe zdjęcia po transformacji
![Transformed Images](https://i.imgur.com/YyehcA2.png)

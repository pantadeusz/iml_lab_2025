### Losowy Las
* **Liczba estymatorów:** 20
* **Dokładność walidacyjna:** 1.0
* **Rozmiar:** ~41 bajtów

### Sieć Neuronowa
#### Początkowa sieć:
* **Architektura**:
    * **Dense:** 128, ReLU
    * **Dense:** 64, ReLU
    * **Dense:** 3, Softmax
* **Optimizer:** Adam (współczynnik uczenia: 0.001)
* **Funkcja straty:** Sparse Categorical Crossentropy
* **Rozmiar:** ~144 bajtów

Trening odbywał się przez 300 epok na rozmiarze batcha równym 32.

> Dokładność walidacyjna: ~0.917

![NN Training History](https://i.imgur.com/G5yU08y.png)

#### Po dodaniu początkowej warstwy normalizującej cechy
W tym przypadku zmniejszona została liczba epok do **100** ze względu na to, że po dodaniu wartswy normalizującej model zaczął dużo szybciej się uczyć.

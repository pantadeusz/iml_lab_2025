### **Losowy Las**
* **Liczba estymatorów:** 20
* **Dokładność walidacyjna:** 1.0
* **Rozmiar:** ~41 kilobajtów

### **Sieć Neuronowa**
#### Początkowa sieć:
* **Architektura**:
    * **Dense:** 128, ReLU
    * **Dense:** 64, ReLU
    * **Dense:** 3, Softmax
* **Optimizer:** Adam (współczynnik uczenia: 0.001)
* **Funkcja straty:** Sparse Categorical Crossentropy
* **Rozmiar:** ~144 kilobajtów

Trening odbywał się przez 300 epok na rozmiarze batcha równym 32.

> Dokładność walidacyjna: ~0.917

![NN Training History](https://i.imgur.com/G5yU08y.png)

#### Po dodaniu początkowej warstwy normalizacyjnej cechy
W tym przypadku zmniejszona została liczba epok do max **150**, różnie dla poszczególnych modeli, gdyż niektóre, mniej skomplikowane potrzebowały więcej czasu na nauczenie. Zostało to zrobione ze względu na to, że po dodaniu wartswy normalizacyjnej model zaczął dużo szybciej się uczyć.

| Units 1 | Units 2 | Rozmiar (Bajty) |
| :--- | :--- | :--- |
| **20** | **6** | **33** |
| 25 | 5 | 34 |
| 28 | 8 | 36 |
| 32 | 8 | 37 |
| 32 | 16 | 40 |
| 48 | 16 | 46 |
| 128 | 16 | 74 |
| 128 | 32 | 99 |
| 128 | 64 | 148 |

**Historia dokładności treningowej i walidacyjnej w przypadku modelu o najmniejszym rozmiarze.**
![Training with normalization](https://i.imgur.com/3LlJjOp.png)

**Historia tego samego modelu po dodaniu regularyzacji L2 w pierwszej i drugiej warstwie.**
![Training with regularization](https://i.imgur.com/Vk9PDGe.png)

> Udało się znacznie wygładzić, ustabilizować trening dzięki warstwie normalizacyjnej oraz regularyzacji L2. 

> Poszukiwania modelu sieci neuronowej o takiej samej 100% dokładności walidacyjnej, ale o mniejszym rozmiarze zakończyły się sukcesem. Udało się znaleźć model o rozmiarze 33 kilobajtów.

>Wartości poszczególnych dokładności walidacyjnych testowanych powyższych modeli wachały się cały czas pomiędzy 94-100%.

**Weryfikujący zrzut ekranu.**

![Screen of models](https://i.imgur.com/E8jVoX2.png)

**Wnioski**

Udało się znaleźć model o liczbie unitów w pierwszej warstwie 20, drugiej 6, o rozmiarze 33 kilobajtów, który okazał tak samo skuteczny jak RandomForest o rozmiarze 41 kilobajtów i 20 estymatorach. Ten model sieci neuronowej ma mniejszy rozmiar równy 33 kB, czyli o 8 kB mniejszy niż lasu losowego. Dodatkowo regularyzacja doprowadziła do stabilniejszego uczenia oraz normalizacja do szybszego i wygładzonego uczenia.
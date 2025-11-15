## Pliki wykorzystane w projekcie
* `main.py` - główny plik odpowiedzialny za znalezienie najlepszej architektury modelu oraz zapisanie go do formatu .keras
* `infer.py` - plik wykonujący inferencję modelu. Model wraz ze zdjęciem są pobierane ze ścieżek podanych w argumentach programu
* `get_images.py` - odpowiada za pobranie 3 zdjęć, każdego posiadającego inną klasę ze zbioru tensorflow datasets 'beans'

## Sieć Neuronowa
Model sieci neuronowej został znaleziony i zoptymalizowany przy użyciu Keras Tuner metodą random search.

### Architektura i Strojenie Hiperparametrów
* **Model:** Sieć neuronowa trójwarstwowa
* **Liczba prób:** Wypróbowano 100 różnych kombinacji modeli.
* **Funkcja kosztu:** SparseCategoricalCrossentropy
* **Optymalizatory:** Adam, AdamW, RMSprop, SGD
* **Inicjalizatory wag:** GlorotUniform, HeNormal, HeUniform, LecunNormal
* **Funkcje aktywacji:** ReLU, Tanh, Sigmoid, eLU, SeLU
* **Epoki:** Każdy model był trenowany przez maksymalnie 100 epok.
* **Batch:** 32
* **Wczesne zatrzymanie:** Zastosowano `EarlyStopping` z `patience=10`. Trening był przerywany, jeśli dokładność walidacyjna nie poprawiła się przez 10 kolejnych epok.
* **Zakres losowania neuronów w poszczególnych warstwach:**
    * **Warstwa 1:** [64, 128]
    * **Warstwa 2:** [32, 128]
    * **Warstwa 3:** [4, 32]
* **Współczynnik uczenia:** [0.0001, 0.01]
> Czas poszukiwań odpowiedniego modelu wyniósł 1h 12min 19s

### Najlepsze znalezione hiperparametry:
* **Layer 1 Units:** 104
* **Layer 2 Units:** 44
* **Layer 3 Units:** 16
* **Optimizer:** Adam
* **Initializer:** HeNormal
* **Activation Function:** ReLU
* **Learning Rate:** ~0.0001

### Podsumowanie modelu
Modelowi udało się uzyskać dokładność na zbiorze walidacyjnym wynoszącą ~0.774.

| Layer (type) | Output Shape | Param # |
| :--- | :--- | :--- |
| `flatten` (Flatten) | (None, 49152) | 0 |
| `dense` (Dense) | (None, 104) | 5,111,912 |
| `dense_1` (Dense) | (None, 44) | 4,620 |
| `dense_2` (Dense) | (None, 16) | 720 |
| `dense_3` (Dense) | (None, 3) | 51 |
**Liczba parametrów:** 5,117,303 (19.52 MB)

## Porównanie parametrów
![Basic Graph](https://i.imgur.com/i5ExSkl.png)
![Advanced Graph](https://i.imgur.com/HHkvSp3.png)
Na wykresach zaprezentowane zostały wyniki median dokładności walidacyjnych osiąganych podczas użycia poszczególnych parametrów. Na zamieszczonych wykresach najlepszą funkcją aktywacji była eLU, najlepszym optimizerem SGD, zaś initializerem HeUniform. 

Można również zauważyć, że połączenie specyficznych funkcji optymalizujący oraz initializerów okazywało się być znaczące. Na przykład tanh w połączeniu z SGD dawał znacznie lepszą dokładność niż pozostałe optimizery (widać również, że występowało mało elementów odstających w tym przypadku) albo HeNormal initializer wraz z Adamem jako optimizerem.

## Wnioski
> Modelowi trójwarstwowej sieci neuronowej udało się uzyskać dokładność ~0.774 na zbiorze walidacyjnym

> Widać różnicę pomiędzy używaniem poszczególnych parametrów. Przykłądowo RMSprop wypadał najgorzej, zaś SGD najlepiej, AdamW niewiele lepiej niż Adam.

> Podobnie było z funkcjami aktywacji. Najlepsze okazały się być eLU, SeLU oraz ReLU, dosyć znacząco w porównaniu z Tanh czy sigmoid.

> Mniejsze różnice lecz i tak zauważalane wystąpiły jeżeli chodzi o użycie różorakich inicjatorów wag, HeUniform wypadł najlepiej. Najmniej skuteczny okazał się być Lecun initializer.
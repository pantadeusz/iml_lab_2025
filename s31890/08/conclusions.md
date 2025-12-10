## Wyniki
* Base model:
  - Standard MNIST: test loss, test acc:  [0.029227646067738533, 0.9937999844551086]
  - Augmented: test loss, test acc:  [6.511506080627441, 0.26910001039505005]
* Retrained model:
  - Standard MNIST: test loss, test acc:  [0.29238131642341614, 0.9125000238418579]
  - Augmented: test loss, test acc:  [0.28155428171157837, 0.906000018119812]

## Architektura
* Model bazowy trenowany przez keras tuner Hyperband przez 90 prób, max 30 epok na zwykłym MNIST bez augmentacji.
* Model retrenowany to model bazowy retrenowany przez równierz 30 epok na augmentowanym zbiorze MNIST bez żadnych zmian w architekturze.

## Wniąski
* Myślę, że w tym przypadku wyniki mówią same za siebie. Początkowy model uzyskał blisko 100% dokładność na zbiorze standardowym bez augmentacji lecz nie był w zupełności gotowy na walidację augmentowanym zbiorem danych.
* Model retrenowany uzyskał bardzo podobne wyniki na zbiorze augmentowanym co na standardowym. Wynik na zbiorze standardowym o prawie 9% niższy niż modelu bazowego, natomiast wynik na zbiorze augmentowanym nieporównywalnie lepszy. Wyniki modleu można by poprawić używając od początku augmentacji dla zbioru treningowego a następnie używając tunera by przygotować model na rożnorodność danych.

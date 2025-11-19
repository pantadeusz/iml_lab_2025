# Lab 08

Dzisiejsze zadanie jest bardzo mocno oparte o przykład z wykładu. Bardzo zachęcam jednak do częściowo samodzielnego przygotowania rozwiazania.

Jako bazę przyjmujemy

* albo https://www.tensorflow.org/datasets/keras_example 
* albo poprzednie rozwiązania.

Wybór jest, ponieważ już coś podobnego trenowaliśmy i można wykorzystać swój wcześniejszy kod.

Zadanie polega na tym, aby zwiększyć dokładność predykcji dla większej liczby przykładów.

## 0

Zanotuj jak uczysz modele - liczba epok, rozmiar batch-a, algorytm uczenia. Te parametry mają być dla każdego uczenia modelu. Może się zmieniać learning rate.

## 1

Przyjmij jako baseline model wygenerowany albo Twoją dawniejszą metodą, albo wytrenowany na bazie przykładu z dokumentacji. Nie staraj się poprawiać - co wyszło to wyszło. Pamiętaj o zapisaniu modelu.

Dodaj możliwość ponownej ewaluacji modelu dla nowych danych, to znaczy, przygotuj funkcję która dla danego modelu i danego zbioru danych (czy to np_array, czy dataset) obliczy metryki.

## 2

Teraz przygotujemy mechanizm augmentacji (wzbogacania) danych. Zobacz https://www.tensorflow.org/tutorials/images/data_augmentation gdzie jest opisane jak można zwiększać różnorodność danych.

Nas interesuje przynajmniej:

* losowy negatyw
* losowy niewielki obrót
* losowe niewielkie przesunięcie

Weź pod uwagę rozdzielczość obrazków - nie można, na przykład, przesuwać więcej niż o kilka pikseli, bo inaczej będą nam obrazki uciekały za margines.

Niech to będzie funkcja - będziemy z niej jeszcze korzystać.

Sprawdź czy działa - albo zapisz kilka losowych obrazków do plików (matplotlibem lub inaczej), albo je wyświetl.

## 3

Zastosuj augmentację do zbioru testowego i użyj go do oceny modelu "baseline". Zanotuj jakie miał wyniki.

## 4

Naucz nowy model na danych z augmentacją. Zapisz go - będzie to nasz model o architekturze bazowego, ale nauczony na danych wzbogaconych. Oceń go stusjąc tą samą metodę co z poprzedniego punktu.

## 5

Stwórz nowy model - musi mieć te samo wejście i wyjście. Zastosuj warstwy konwolucyjne, max pooling i w pełni połączone. Możesz polegać na intuicji - nie będziemy robili optymalizacji hiperparameterów - nie zdążymy.

## 6

Przetrenuj nowy model i oceń go na zbiorze testowym, tak samo jak w przypadku poprzednich modeli. Zapisz wyniki.

## 7

Już tradycyjnie - proszę o krótkie podsumowanie. Porównaj jak zachowały się modele w zależności od tego, na czym uczyliśmy oraz przy zastosowaniu różnych architektur.

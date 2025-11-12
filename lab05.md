# Lab 5

Na wykładzie powiedzieliśmy sobie o tym, że można dostosowywać parametry uczenia do tego, aby uzyskać jak najlepsze efekty. Na dzisiejszych ćwiczeniach zastosujemy autotuner. Na wykładzie wspomniałem o tym narzędziu.

## Zadanie 1

Upewnij się, że Twoja metoda na ładowanie i uczenie modeli (wersja TensorFlow i Scikit) działa dla zbioru danych który masz wybrane. Jeśli są problemy - proszę popraw.

Sugestie na co warto zwrócić uwagę:

* Ładowanie danych
* Trenowanie bez błędów
* Rozsądne predykcje

## Zadanie 2

Upewnij się, że cały kod jest podzielony na poszczególne funkcje, ponieważ teraz będziemy starali się dostosować DNN tak, aby dawała lepsze wyniki niż rozwiązanie oparte o Scikit. Musi być widać jakie funkcje pracują na naszym modelu bazowym (baseline) a jakie na tym opartym o głębokie sieci neuronowe.

Przy okazji tego zadania możesz dodać 1-2 warstwy JEŚLI masz pewność co robisz, inaczej zostaw jak jest. Zachęcam do zadawania pytań, jeśli nie wiesz co robisz.

Niech po tym etapie skrypt będzie pokazywał (na konsoli) wyniki walidacji dla obu modeli - dodatkowo macierz pomyłek i miary liczone przez scikit-learn classification_report.

## Zadanie 3

Wepnij autotuner (Keras Tuner) do Twojego kodu. Najprawdopodobniej będzie trzeba dostosować kod tworzenia modelu, ale może nie u wszystkich. Warto zapisywać powstałe modele (zob. ```tuner.get_best_models()``` oraz dokumentację <https://keras.io/keras_tuner/getting_started/>)

Podpowiedź/sugestia dotycząca budowania modelu:

```python
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu'))
    # ...
    return model
```


## Zadanie 4

Eksperymentuj z autotunerem i dostosowaniem modelu DNN tak, aby uzyskał lepszy wynik niż rozwiązanie oparte o Scikit. Jeśli się nie uda, to proszę wykazać jaki był proces osiągnięcia tej "porażki".

Wyjaśnienie - porażka w cudzysłowiu, ponieważ tak na prawdę nie jest to porażka o ile eksperyment został przeprowadzony rzetelnie.

Po upewnieniu się, że Keras Tuner jest zastosowany (można go uruchomić i nie zgłasza błędów składni), zacznij eksperymenty.

Zasoby i ograniczenia:

* Proszę na te eksperymenty poświęcić maksymalnie 30 minut. Co wyjdzie, będzie OK.
* W tym zadaniu pracujemy tylko z tunerem, nie zmieniamy architektury sieci
* Przed uruchomieniem większego eksperymentu oszacuj ile czasu zajmie liczenie - można policzyć czas pojedynczego uczenia pomnożony przez liczbę kombinacji parametrów

## Zadanie 5

Opisz proszę eksperymenty i wyniki w Markdown. Należy uwzględnić:

* Ogólny opis architektury - liczba i rozmiary warstw, funkcje aktywacji
* Ogólny opis eksperymentu
* Parametry autotunera
* Wyniki - classification_report dla obu finalnych modeli - baseline oraz zoptymalizowany oparty o sieć neuronową.
* Twoje wnioski (jeden akapit)

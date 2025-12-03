# Lab 10 - Rekurencyjne sieci neuronowe

Dzisiaj będziemy bazowali na samouczku od autorów TensorFlow dotyczącym analizy sentymentu.

Oryginalny tutorial jest https://www.tensorflow.org/text/tutorials/text_classification_rnn

Musiałem go zaktualizować - chodziło o typy danych do metody predict. Wersja zaktualizowana jest 
w docs/tutorials/text_classification_rnn.ipynb

## Zadanie 1

Zapoznaj się z samouczkiem - jeśli coś jest do wyjaśnienia - KONIECZNIE ZAPYTAJ.

Postaraj sie uruchomić lokalnie program z tego tutoriala - można wyeksportować jako kod.

Jeśli działa u Ciebie lokalnie lub na maszynie zdalnej z procesorem do obliczeń - to jest OK.

## Zadanie 2

Uporządkowanie kodu. Wiadomo - funkcje i klasy.

Cel:

program będzie uczył się sentymentu i finalnie zapisze model do pliku.

## Zadanie 3

Stwórz drugi program, który będzie ładował model z pliku, a ze standardowego wejścia będzie odbierał tekst. Po zakończeniu strumienia wejściowego wyświetli na terminalu predykcję - czy jest to pozytywny czy negatywny komentarz.

Podpowiedź:

```
import sys

for line in sys.stdin:
    print(line)
```

## Zadanie na dodatkowy 0.5pkt

Zastosuj Keras Tuner albo inną metodę optymalizacji hiperparametrów. Postaraj się wytrenować jak najlepszy model predykcji sentymentu.

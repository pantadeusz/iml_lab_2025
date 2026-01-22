# Lab 09 – Autoencoder

Celem dzisiejszych zajęć jest automatyczne prostowanie obrazków ze zbioru Fashion MNIST za pomocą autoenkodera.

## Zadanie 1

Przygotuj środowisko – proszę oprzeć się o przykład dotyczący autoenkodera z [TensorFlow Autoencoder Tutorial](https://www.tensorflow.org/tutorials/generative/autoencoder).

Uruchom ten przykład lokalnie oraz na maszynie z GPU do obliczeń. Interesuje nas etap przed sekcją [Second Example: Image Denoising](https://www.tensorflow.org/tutorials/generative/autoencoder#second_example_image_denoising) (nie robimy odszumiania).

## Zadanie 2

Dodaj augmentację polegającą na niewielkim obracaniu obrazków. Na wyjściu chcemy uzyskać obrazki bez obrotu (prostowanie).

## Zadanie 3

Dodaj warstwę lub warstwy konwolucyjne na wejściu. Sprawdź, jak zmieniają się wyniki.

## Zadanie 4

Finalny program powinien zapisywać modele (enkoder i dekoder) do plików.

Dodaj dodatkowy program, który będzie ładował wytrenowany autoenkoder i dla zadanego obrazka na wejściu zapisywał wynikowy obrazek na wyjściu oraz wyświetlał na konsoli wektor ukryty (latent).

# Analiza badania czułości i swoistości modelu w predykcji probalistycznej #

## Wyniki ##
Wyniki z predykcji probalistycznej

**Sensitivity** - odsetek poprawnie wykrytych przypadków pozytywnych

**Specificity** - odsetek poprawnie wykrytych przypadków negatywnych

    Threshold = 0 (same 1)
        Sensitivity 1.0
        Specificity 0.0

    Threshold = 0.1
        Sensitivity 0.99
        Specificity 0.88

    Threshold = 0.2
        Sensitivity 0.99
        Specificity 0.91

    Threshold = 0.3
        Sensitivity 0.99
        Specificity 0.91 

    Threshold = 0.4
        Sensitivity 0.99
        Specificity 0.91

    Threshold = 0.5
        Sensitivity 0.99
        Specificity 0.91 

    Threshold = 0.6
        Sensitivity 0.99
        Specificity 0.95

    Threshold = 0.7
        Sensitivity 0.99
        Specificity 0.95

    Threshold = 0.8 
        Sensitivity 0.97
        Specificity 0.98

    Threshold = 0.9
        Sensitivity 0.90
        Specificity 1

    Threshold = 1 (same 0)
        Sensitivity 0.0
        Specificity 1.0

## Co się dzieje gdy zmieniamy próg (threshold)? ##

### Niski threshold (np. 0.1) ###
* Wysoka czułość
* Niska swoistość
* Model wyłapuje więcej przypadków pozytywnych, ale częściej robi fałszywych alarmów


**Może się do przydać do priorytetyzacji przypadków fałszywych pozytywnych niż fałszywych negatywnych np. dla modeli 
związanym ze zdrowiem**  

### Wysoki threshold (np. 0.9) ###
* Model wymaga bardzo dużego prawdopodobieństwa, by uznać 1.
* Wysoka swoistość, mało fałszywych alarmów.
* Niska czułość, częściej pomija prawdziwe pozytywne przypadki.

## Wnioski ##
* **Im wyższy** próg (threshold), tym mniej wyników klasyfikowanych jako pozytywne.
* Wzrasta swoistość (TN / (TN + FP)), czyli model rzadziej daje fałszywe alarmy.
* Spada czułość (TP / (TP + FN)), czyli model częściej przegapia przypadki pozytywne.

W zależność od danego problemu wybieramy odpowiedni próg tak, aby model lepiej spełniał założenia projektowe.
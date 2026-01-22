# Podsumowanie - lab 6

Po wielu próbach udało się osiągnąc accuracy na poziomie 75% ze stratą
około 0.64. 

Dwa najlepsze modele - jeden zrobiony w pełni manualnie, a drugi z tunerem -
korzystały z aktywacji `elu`. Kolejne lepsze modele korzystały z aktywacji `relu`,
ale warto zwrócić uwagę na model z równie dobrym wynikiem, który używał `tanh`
i powstał korzystając z tunera.

Wszystkie najlepsze tunerowe modele wybrały optymalizator `SGD` włącznie z najlepszym
ogólnie modelem i tym jednym z `tanh`, natomiast w manualnych próbach `adam`
również osiągał dobre wyniki.

#### Parametry modelu z najlepszym wynikiem:
Warstwy:
- flatten
- elu 80
- elu 120
- elu 272
- elu 464
- softmax 3

Optimizer: SGD

# Podsumowanie wyników modelów o różnych rodzajach imputacji #

Do badania wyników wykorzystałem ROC AUC
(AUC - jest to pole pod krzywą. Pokazuje, jak dobrze model odróżnia klasy niezależnie od progu. Im bliżej 1 tym bardziej precyzyjniejszy)

## Wnioski ##
- Braki danych zmniejsza skuteczność modelu
- Przy dużym odsetków braków np. 90% wszystkie metody imputacji znacznie spadają
- Przy 20% brakach - metody imputacji działają lepiej niż bez braków.
- Porównując wyniki z braków od 10% do 90% można zauważyć, że Mean i MICE mają mniejszą tendencje spadkową niż KNN.


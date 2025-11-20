# Wnioski do zadania 1

## Braki 5-10%
MICE dał najdokładniejsze rezultaty, które są najbliższe modelowi referencyjnemu. 
- Dla 5% braków MICE osiągnął MSE wynoszący 2973 (+73 modelu referencyjnego), podczas gdy reszta metod stanowczo odbiega od tej normy – Mean miał MSE 3245 (+345), a KNN 3130 (+230).
- Dla 10% braków MICE osiągnał MSE 2985 (+85), Mean 3118 (+218), KNN 3143 (+243).

## Brak 20%
Na poziomie braku 20% można zaobserwować anomalię. Wszystkie modele mają większą dokładność niż model referencyjny.
- Mean 2827 (-73)
- KNN 2880 (-20)
- MICE 2842 (-58)

Może to wynikać z tego, że usunięte zostały akurat takie dane, które najbardziej odbiegają od średniej.

## Brak 30%
Przy 30% braków różnica między metodami znacząco się zmniejszyła.
- Mean 2931 (+31)
- KNN 2948 (+47)
- MICE 2972 (+72)

MICE, który do tego momentu był najdokładniejszy, wypadł najgorzej. Powodem może być to, że MICE jest najbardziej zaawansowaną metodą spośród trzech testowanych, która przewiduje braki iteracyjnie. W tym przypadku mamy tylko 70% danych wejściowych, stąd może wynikać ta trudność.

### Dlatego wybór metody powinien zależeć od poziomu braków. Dla małych MICE jest oczywistym wyborem, a dla dużych KNN/Mean.

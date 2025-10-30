# Sieć neuronowa - wnioski

Porównanie modeli **Random Forest** i prostej **sieci neuronowej**
w klasyfikacji binarnej na zbiorze `load_breast_cancer()`

W czasie testowania wielokrotnie zmieniałem ilość warstw
ukrytych, jak i neuronów w danych warstwach. Po wielu próbach
doszedłem do wniosku, że optymalna ilość warstw ukrytych to 
2-3 warstwy, w zależności od liczby neuronów. 

Jeżeli zaczynamy od dużej ilości neuronów (np. 64) to trzeba
przynajmniej raz użyć drop-outu, w innym wypadku model się przeucza.

Model **Random Forest** w większości prób otrzymywał lekko lepsze
wyniki niż model **sieci neuronowej**, jednak po dobraniu odpowiedniej
liczby warstw ukrytych i neuronów w tych warstwach udało się kilka razy
osiągnąć dokładniejsze wyniki.

Wychodzi na to, że to jaki model warto użyć w praktyce mocno zależy
od zadania. W przypadku klasyfikacji binarnej model **Random Forest**
jest dobrym wyborem, jednak jeżeli chodzi o dane zdrowotne, może
warto poświęcić czas i złożoność związaną z siecią neuronową, aby
otrzymać lepsze wyniki.
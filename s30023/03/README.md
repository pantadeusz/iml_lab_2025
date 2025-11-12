# Zadanie 1

- Przy zmianie procentu braków danych [5%, 10%, 20%, 30%, 40%] zauważyłem, że dla 5% i 10% wyniki funkcji PR są gorsze.
- Modele KNN i MEAN osiągają bardzo podobne wyniki.
- Model MICE radzi sobie gorzej przy większym braku danych, natomiast lepiej przy mniejszym (w porównaniu do KNN i MEAN).

# Zadanie 2

## Róznica między modelami

- Base - model bazowy, wytrenowany na oryginalnych danych
- Weighted - model, który automatycznie nadaje większą wagę rzadziej występującej klasie
- SMOTE - model, w którym technika SMOTE generuje syntetyczne przykłady klasy mniejszościowej na podstawie istniejących danych
- Under - model, który usuwa część przykładów z klasy dominującej

## Zmiana progu

- Base: dobre wyniki tylko przy bardzo niskim progu (10%).
- Weighted: najlepszy kompromis przy progu +- 70% (F1 = 0.41).
- SMOTE: również najlepsze wyniki przy progu ok. 0.7.
- Under: działa gorzej niż SMOTE i Weighted, ale lepiej niż bazowy.
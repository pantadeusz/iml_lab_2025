# Radzenie sobie z brakującymi danymi - wnioski

Model był testowany dla braków danych o wielkości od 5% do 40%, 
zmieniając wartość co 5%, oraz jednorazowo dla 60%.

Dla braków danych od 5% do około 20%-25% R^2 wzrastało, 
jednak po przekroczeniu tej granicy zaczyna maleć. 
Warto zaznaczyć, że model nigdy nie osiągnął R^2 na 
poziomie wyższym niż 50%. Dla braków danych wynoszących 60% 
zbioru R^2 drastycznie spada, zwłaszcza dla imputacji 
metodą **KNN**.

RMSE osiąga najmniejszą wartość również w okolicach 20%-25%, 
utrzymując pozostałe wartości na podobnym poziomie. 
Nieznaczny wzrost można zauważyć dopiero przy 35% i wyżej, 
jednak nie jest on tak duży, jak w przypadku R^2.

Różne metody imputacji dają bardzo podobne wyniki 
dla średnich wartości (10%-30%) braków danych, 
jednak wyraźnie różnią się dla wartości skrajnych 
(5% oraz 35%-60%). Najlepszym poziomem wykazała się 
metoda **KNN** przy 20%, lecz jeśli braki występują przy 
wartościach skrajnych, lepiej działa metoda **Mean**.
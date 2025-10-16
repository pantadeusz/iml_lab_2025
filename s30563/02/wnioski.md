# Podsumowanie

Im większy jest próg decyzyjny, tym lepsza jest **swoistość**,
natomiast, gdy zmniejszy się go wzrasta **czułość**.

W przypadku danych z `load_brest_cancer()` **czułość** oraz
**swoistość** utrzymują się przy podobnych wartościach dla
progu decyzyjnego od **_18%_** do **_55%_**. 

Dla wartości do **_83%_** **swoistość** znacząco rośnie małym 
kosztem czułości, osiągając **_100%_** poprawnie 
wykrytych przypadków negatywnych, utrzymując **_97%_** 
poprawnie wykrytych przypadków pozytywnych.

Jeżeli pójdziemy w drugą stronę **czułość** osiąga
**_100%_** przy progu decyzyjnym na poziomie **_8%_**
z niewielkim spadkiem (około **_2%_**) **swoistości** 
względem początkowego progu (**_50%_**). **Swoistość** 
znacząco spada dopiero przy wartościach progu od **_1%_**.


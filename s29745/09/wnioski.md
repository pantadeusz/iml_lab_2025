#### Model bazowy
Model z dokumentacji osiąga loss na zbiorze treningowym równy 0.0160, natomiast na walidacyjnym 0.0161.

#### Konwolucje
Po dodaniu warstw konwolucyjnych już przy 10 epokach model poprawił wyniki uzyskując na zbiorze treningowym loss na poziomie 0.0104, a na walidacyjnym 0.0109. Natomiast po zwiększeniu liczby epok do 30 model na zbiorze treningowym uzyskał loss na poziomie 0.0088, a na walidacyjnym 0.0102.

#### Wnioski
oba autoencodery nieźle radzą sobie z przekształceniami i raczej uogólniają obrazy nie zachowujac ich szczegółów, a jedynie kształt. Dodatkowo wynik po autoencoderze konwolucyjnym zdaje się mniej zaszumiony.
Oczekiwane zachowanie: im mniejszy próg, tym wyższa czułość i niższa swoistość; im większy próg, tym odwrotnie.

no i dokładnie tak sie dzieje, widać ze model po zwiększeniu 
threshold np. na 0.9 juz ma swoistosc 1.0 i nie ma ani jednego
przypadku FN i w drugą strone to samo, myślę ze model jest
najlepiej wywazony przy 0.70 lub 0.75 bo wtedy pokazuje 
najmnniejsza ilosc F, gdzie przy 0.70 pokazuje 2xFN i 1xFP a przy 0.75 dokładnie odwrotnie, 
nie da rady zejść nizej
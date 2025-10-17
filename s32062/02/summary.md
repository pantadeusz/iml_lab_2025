# Podsumowanie wpływu zmiany progu decyzji na precyzję i swoistość klasyfikatora.

Zwiększanie progu decyzji sprawia, że precyzja i swoistość rosną, a jego zmniejszanie, że maleją. Wartość swoistości, niezależnie od klasy, zmienia się w przedziale od 0 do 1, natomiast wartość precyzji od ok. 0.62 do 1 dla klasy '1' i od ok. 0.37 do 1 dla klasy '0'. Gdy wartość progu decyzji jest skrajnie pozytywna, pojawia się błąd przy próbie obliczenia precyzji, ponieważ dzielona jest liczba rzeczywista przez 0.

Badając zmianę progu decyzyjnego można znaleźć różnicę między jego mniejszą a większą wartością

Jeżeli ustawimy go na mniejszą wartość (np. 0.3), więcej danych będzie zakwalifikowanych jako 1 (komórka chora). Jest jednak szansa, że TP i NP znacznie wzrosną, gdyż więcej komórek (w tym zdrowych) zostanie uznanych jako chore.

W tym wypadku czułość znacznie wzrośnie, gdyż model coraz rzadziej pomija dane, lecz swoistość maleje, gdyż model nie jest w stanie wykryć fałszywych alarmów.

Jeżeli ustawimy próg decyzyjny na większą wartość (np. 0.8), model jest bardziej restrykcyjny. Częściej wyłapujemy zdrowe przypadki i uniknie robienia błędów.
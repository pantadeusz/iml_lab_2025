# Wnioski z twierdzenia Bayesa #

## Wpływ częstości choroby (prior) ##
- Jeśli choroba jest rzadka (np. 0.1%), nawet bardzo dobry test może dawać dużo fałszywie pozytywnych wyników.
- Im częstsza choroba, tym wyższa pewność pozytywnego wyniku.

**Wniosek**: Rzadkie choroby wymagają dodatkowej weryfikacji, nawet przy dobrych testach.

## Wpływ swoistości (specificity) ##
- Niższa swoistość - więcej fałszywie pozytywnych wyników → posterior spada.
- Wyższa swoistość (0.99) - wynik Test+ jest bardzo wiarygodny, szczególnie przy rzadkich chorobach.

**Wniosek**: Aby test dobrze przewidywał chorobę, szczególnie rzadką, swoistość jest kluczowa.

## Sekwencja testów ##
- Kolejne testy (przy wysokiej czułości i swoistości) drastycznie zwiększają posterior.
- Jeśli wynik pierwszego testu jest pozytywny, następny test potwierdza diagnozę i zmniejsza ryzyko fałszywego alarmu.

**Wniosek**: Dla rzadkich chorób lub przy niepewnych testach warto stosować testy sekwencyjne, aby zwiększyć pewność diagnozy.
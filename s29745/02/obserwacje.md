# Obserwacje i wnioski w ramach eksperymentowania z progiem decyzji

## Dla progu 0.0
Czułość: 1.0
Swoistość: 0.0

## Dla progu 0.1
Czułość: 0.986
Swoistość: 0.883

## Dla progu 0.2
Czułość: 0.986
Swoistość: 0.907

## Dla progu 0.4
Czułość: 0.987
Swoistość: 0.907

## Dla progu 0.6
Czułość: 0.986
Swoistość: 0.953

## Dla progu 0.8
Czułość: 0.972
Swoistość: 0.977

## Dla progu 0.9
Czułość: 0.887
Swoistość: 1.0

## Dla progu 1.0
Czułość: 0.0
Swoistość: 1.0

## Wnioski:
- Zmiana progu decyzji wpływa na kompromis między czułością a swoistością modelu.  
- Niższy próg zwiększa czułość, ale zmniejsza swoistość, natomiast wyższy próg działa odwrotnie.  
- Optymalne wartości progu leżą zwykle w okolicach 0.5–0.6, gdzie model zachowuje równowagę między wykrywaniem pozytywów a unikaniem fałszywych alarmów.  
- Przy progach skrajnych model staje się praktycznie bezużyteczny, przewidując albo wszystko jako pozytywne, albo nie przewidując żadnego pozytywnego przypadku.

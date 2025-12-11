# Wnioski do zadania 2

## Ocena modeli z podstawowym threshold=0.5
### Skupiamy się na klasie 1, gdyż jest ona mniejszościowa i kluczowa do wykrycia

| Method   | Precision | Recall | F1   |
|----------|-----------|--------|------|
| Base     | 0.33      | 0.08   | 0.13 |
| Weighted | 0.18      | 0.83   | 0.29 |
| SMOTE    | 0.19      | 0.75   | 0.30 |
| Under    | 0.13      | 0.58   | 0.22 |

- Base model - wyróżnia się najlepszą precyzją, mimo to osiągnął tragiczny Recall i zarazem F1-score. Bierze się to z tego powodu, że operuje on na bardzo niezbalansowanych klasach (95% klasa 0; 5% klasa 1). 
- Weighted model - wykrywa, aż 83% przypadków klasy 1, ale za cene wielu fałszywych alarmów (0.18)
- SMOTE model - zbliżone wyniki do weighted model, ale wykrywa więcej przypadków klasy 1 co wpłynęło na F1-score lepszy o 0.01
- Under model - poradził sobie najgorzej z 3 balansujacych modeli

### Najlepiej poradził sobie model z balansowaniem SMOTE (F1-score=0.30). Model z balansowaniem ważonym, jednak osiągnął bardzo zbliżone wyniki (F1-score=0.29)

## Wpływ progowania na wyniki 

- Niski threshold = Więcej predykcji klasy 1 = Recall rośnie, Precision maleje
- Wysoki threshold = Mniej predykcji klasy 1 = Recall maleje, Precision rośnie

### Base model
- Po zmniejszeniu progu decyzyjnego F1-score wyraźnie zmalał (0.13 -> 0.29), czyli prawie taki sam wynik jak nasz najlepszy model z balansowaniem przy podstawowym progu.

### Weighted model
- Po zmniejszeniu progu decyzyjnego F1-score zmalał, ale warto zauważyć, że przy progu 0.1 osiągneliśmy Recall 1.00
- Po zwiększeniu progu decyzjnego F1-score urósł, aż do rekordowego 0.37

### SMOTE model
- Po zmniejszeniu i zwiększeniu progu decyzyjnego F1-score zmalał

### Under model
- Po zmniejszeniu progu decyzyjnego F1-score zmalał
- Po zwiększeniu progu decyzyjnego F1-score urósł, jednakże najlepszy wynik osiągnął na progu 0.7

### Najwyższy F1-score dał model z balansowaniem ważonym przy progu 0.8

### Za pomocą progowania udało się uzyskać lepsze wyniki dla każdego modelu oprócz SMOTE, który najlepsze wyniki osiągał przy podstawowym progu. Jest to przydatne narzędzie, które może znacząco poprawić dokładność naszego modelu. Mimo, że F1-score jest najbardziej miarodajną metryką, to wybór balansowania modelu czy progu decyzjnego, w głównej mierze powinien zależeć od naszego problemu. 
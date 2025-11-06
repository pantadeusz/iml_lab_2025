### Wyniki:
Original: PR-AUC = 0.8940
SMOTE: PR-AUC = 0.8976
Undersampling: PR-AUC = 0.9049
Class Weights: PR-AUC = 0.8927

### Wniąski:
Wszystkie metody lekko poprawiły wydajność modelu, szczególnie pod względem krzywej Precision-Recall (PR).
Wyniki PR-AUC były bardzo zbliżone (0,89–0,90), co wskazuje na skuteczność wszystkich podejść.

Zachowanie krzywych:
Precyzja pozostawała wysoka do poziomu przeciętnego recall (0,8),  
po tym punkcie zaczęła spadać — co oznacza, że model był bardzo pewny w swoich wczesnych przewidywaniach.

Ciekawym zjawiskiem wykazało się ważenie klas które straciło więcej skuteczności na początek nawet w porównaniu do oryginału,
lecz z wyższym poziomem recall zachowało jej najwięcej.


Oto przykładowe wyniki dla różnych wartości progu decyzji:

Threshold set to 0.1
{'1': precision: 0.9333333333333333, specificity: 0.8837209302325582  recall: 0.9859154929577465, f1_score: 0.9589041095890412, support: 71"}

Threshold set to 0.3
{'1': precision: 0.9459459459459459, specificity: 0.9069767441860465  recall: 0.9859154929577465, f1_score: 0.9655172413793103, support: 71"}

Threshold set to 0.5
{'1': precision: 0.9459459459459459, specificity: 0.9069767441860465  recall: 0.9859154929577465, f1_score: 0.9655172413793103, support: 71"}

Threshold set to 0.7
{'1': precision: 0.9722222222222222, specificity: 0.9534883720930233  recall: 0.9859154929577465, f1_score: 0.979020979020979, support: 71"}

Threshold set to 0.9
{'1': precision: 1.0, specificity: 1.0  recall: 0.9014084507042254, f1_score: 0.9481481481481481, support: 71"}


czułość - prawdopodobieństwo że chory pacjent zostanie prawidłowo sklasyfikowany.

specyficzność - prawdopodobieństwo że zdrowy pacjent zostanie prawidłowo sklasyfikowany

Wraz z podnoszeniem progu decyzyjnego specyficzność rośnie natomiast przy bardzo wysokim progu (0.9) czułość spada.

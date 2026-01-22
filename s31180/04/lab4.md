# Porównanie metod uczenia Random Forest vs DNN (prosty model) #

### Wybrałem dataset load_breast_cancer z scikit-learn

## Random Forest
|                | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| **0**          | 0.98      | 0.93   | 0.95     | 43      |
| **1**          | 0.96      | 0.99   | 0.97     | 71      |
| **accuracy**   |           |        | 0.96     | 114     |
| **macro avg**  | 0.97      | 0.96   | 0.96     | 114     |
| **weighted avg** | 0.97    | 0.96   | 0.96     | 114     |


## DNN

|                | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| **0**          | 1.00      | 0.98   | 0.99     | 43      |
| **1**          | 0.99      | 1.00   | 0.99     | 71      |
| **accuracy**   |           |        | 0.99     | 114     |
| **macro avg**  | 0.99      | 0.99   | 0.99     | 114     |
| **weighted avg** | 0.99    | 0.99   | 0.99     | 114     |

## Wnioski
**DNN osiągnął wyraźnie lepsze wyniki niż Random Forest: accuracy wzrosło z 0.96 do 0.99, a wszystkie średnie metryki (macro i weighted) z 0.96–0.97 do 0.99. Random Forest już klasyfikuje bardzo dobrze, ale sieć neuronowa jeszcze lepiej równoważy precision i recall dla obu klas, praktycznie eliminując błędy klasyfikacji.**
# Porównanie: RandomForest (sklearn) vs DNN (TensorFlow/Keras)

## Metryki (zestaw testowy)

| Model | Accuracy | Precision | Recall | ROC AUC |
|---|---:|---:|---:|---:|
| RandomForest | 0.9474 | 0.9583 | 0.9583 | 0.9937 |
| DNN (Keras) | 0.9737 | 0.9859 | 0.9722 | 0.9937 |

## Wnioski
- RandomForest to prosty, mocny baseline dla danych tablicowych, na tym zbiorze zwykle wypada bardzo dobrze.
- Prosty DNN bywa porównywalny lub minimalnie gorszy bez tuningu, poprawa wymaga strojenia lub większej sieci.

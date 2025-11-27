--- Ewaluacja: Baseline (Normal Training) na ds_test_normal ---
Loss: 0.0887, Accuracy: 0.9724

--- Ewaluacja: Baseline (Normal Training) na ds_test_augmented ---
Loss: 15.0778, Accuracy: 0.2774

-- Ewaluacja: Baseline (Augmented Training) na ds_test_augmented ---
Loss: 0.3232, Accuracy: 0.9026

--CNN--
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 26, 26, 32)          │             320 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 13, 13, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 11, 11, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 5, 5, 64)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten_2 (Flatten)                  │ (None, 1600)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_4 (Dense)                      │ (None, 128)                 │         204,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_5 (Dense)                      │ (None, 10)                  │           1,290 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘

--- Ewaluacja: CNN (Augmented Training) na ds_test_augmented ---
Loss: 0.1153, Accuracy: 0.9625

## Podsumowanie wyników

| Model                             | Zbiór testowy | Accuracy  | Loss  |
|-------                            |---------------|---------- |------ |
| Baseline (trening normalny)       | normalny      | 97.24%    | 0.09  |
| Baseline (trening normalny)       | augmentowany  | 27.74%    | 15.08 |
| Baseline (trening z augmentacją)  | augmentowany  | 90.26%    | 0.32  |
| CNN (trening z augmentacją)       | augmentowany  | 96.25%    | 0.12  |

## Wnioski

Model trenowany wyłącznie na czystych danych kompletnie zawodzi na zaugmentowanym zbiorze testowym (spadek z 97% do 27%), co pokazuje brak zdolności do generalizacji. Augmentacja danych podczas treningu znacząco poprawia odporność modelu na transformacje (inwersja kolorów, rotacja, translacja), podnosząc accuracy z 27% do 90%. Sieć konwolucyjna (CNN) osiąga najlepsze wyniki (96.25%), ponieważ jej architektura z filtrami splotowymi i warstwami pooling naturalnie lepiej radzi sobie z transformacjami przestrzennymi.

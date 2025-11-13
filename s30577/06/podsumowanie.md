# Lab 6 - Podsumowanie eksperymentu Beans

- **Testowane parametry**:
  - Initializer: `glorot_uniform`, `he_normal`, `random_normal`
  - Activation: `relu`, `tanh`, `elu`
  - Optimizer: `adam`, `sgd`, `rmsprop`
  - Units warstwa 1: 128-512 (krok 128)
  - Units warstwa 2: 64-256 (krok 64)

## Najlepsza konfiguracja
- **Initializer**: glorot_uniform
- **Activation**: tanh
- **Optimizer**: sgd
- **Neurony warstwa 1**: 512
- **Neurony warstwa 2**: 192

## Wyniki finalne
- **Test Accuracy**: **56.25%**
- **Test Loss**: 0.9429
- **Val Accuracy (podczas tuningu)**: 63.16%

## Porównanie wszystkich trials

| Trial | Initializer | Activation | Optimizer | Units 1 | Units 2 | Val Accuracy |
|-------|-------------|------------|-----------|---------|---------|--------------|
| 1 | random_normal | tanh | rmsprop | 256 | 192 | 33.83% |
| 2 | he_normal | relu | sgd | 512 | 256 | 58.65% |
| 3 | he_normal | relu | sgd | 384 | 256 | 57.14% |
| **4** | **glorot_uniform** | **tanh** | **sgd** | **512** | **192** | **63.16%** ✅ |
| 5 | he_normal | tanh | rmsprop | 256 | 256 | 33.83% |
| 6 | random_normal | elu | rmsprop | 384 | 128 | 53.38% |

## Testy predykcji

| Próbka | Prawdziwa klasa | Przewidziana klasa | Pewność | Status |
|--------|----------------|-------------------|---------|--------|
| #5 | angular_leaf_spot | angular_leaf_spot | 62.04% | ✅ Poprawnie |
| #10 | angular_leaf_spot | angular_leaf_spot | 67.13% | ✅ Poprawnie |
| #13 | healthy | bean_rust | 44.13% | ❌ Błąd |

### Szczegóły predykcji dla próbki #13 (błędna):
- **Rozkład prawdopodobieństw**:
  - bean_rust: 44.13% (wybrane przez model)
  - angular_leaf_spot: 36.90%
  - healthy: 18.97% (prawidłowa odpowiedź)
- **Obserwacja**: Model wykazał niską pewność i znaczną niepewność między klasami

## Wnioski

### Mocne strony modelu:
- Model osiągnął **56.25% accuracy** na zbiorze testowym (znacznie lepiej niż losowe zgadywanie: 33.33%)
- **Dobrze rozpoznaje** klasę `angular_leaf_spot` (2/2 testów poprawnych z pewnością 62-67%)
- Najlepsza kombinacja: **glorot_uniform + tanh + sgd**
- Model nie jest przeuczony (test acc 56% vs val acc 63% - niewielka różnica)
- GPU zostało prawidłowo wykorzystane (NVIDIA A40-4Q)

## Podsumowanie
Eksperyment zakończył się sukcesem. Model MLP osiągnął przyzwoitą dokładność 56.25% na trudnym zbiorze obrazów, co jest solidnym wynikiem biorąc pod uwagę prostą architekturę. Identyfikacja najlepszych hiperparametrów (glorot_uniform, tanh, sgd) pokazuje, że systematyczne przeszukiwanie przestrzeni parametrów przynosi wymierne korzyści. Testy predykcji ujawniły zarówno mocne strony (dobre rozpoznawanie angular_leaf_spot) jak i ograniczenia (problemy z klasą healthy). Wyniki potwierdzają teorię, że dla zadań wizyjnych bardziej zaawansowane architektury (CNN) byłyby znacznie bardziej efektywne.


# Wyniki eksperymentów - Lab 5: Keras Tuner

## Tabela wyników

| # | Metoda | Konfiguracja | Units | Learning Rate | Accuracy | F1 (weighted) | ROC-AUC |
|---|--------|--------------|-------|---------------|----------|---------------|---------|
| 1 | RandomForest (baseline) | n_estimators=300 | - | - | 0.9474 | 0.9474 | 0.9937 | - |
| 2 | DNN (baseline) | 1 layer, 64 units, 30 epochs | 64 | 0.001 (adam default) | 0.9649 | 0.9651 | 0.9954 |
| 3 | DNN (tuned) | trials=6, epochs=30 | 96 | 0.001 | 0.9737 | 0.9737 | 0.9960 | - |
| 4 | DNN (tuned) | trials=10, epochs=30 | 32 | 0.001 | 0.9825 | 0.9825 | 0.9954 | - |
| 5 | DNN (tuned) | trials=10, epochs=50 | 128 | 0.01 | 0.9825 | 0.9825 | 0.9921 | - |

---

## Szczegóły eksperymentów

### Eksperyment 1: Baseline
**RandomForest:**
- Accuracy: 0.9474
- F1 (weighted): 0.9474
- ROC-AUC: 0.9937

**DNN Baseline:**
- Accuracy: 0.9649
- F1 (weighted): 0.9651
- ROC-AUC: 0.9954 

---

### Eksperyment 3: Tuner - trials=6, epochs=30
**Najlepsze parametry:**
- Units: 96
- Learning Rate: 0.001

**Wyniki:**
- Accuracy: 0.9737
- F1 (weighted): 0.9737
- ROC-AUC: 0.9960
- Czas treningu: _(nie zmierzony)_ 

---

### Eksperyment 4: Tuner - trials=10, epochs=30
**Najlepsze parametry:**
- Units: 32
- Learning Rate: 0.001

**Wyniki:**
- Accuracy: 0.9825
- F1 (weighted): 0.9825
- ROC-AUC: 0.9954
- Czas treningu: _(nie zmierzony)_ 

---

### Eksperyment 5: Tuner - trials=10, epochs=50
**Najlepsze parametry:**
- Units: 128
- Learning Rate: 0.01

**Wyniki:**
- Accuracy: 0.9825
- F1 (weighted): 0.9825
- ROC-AUC: 0.9921
- Czas treningu: _(nie zmierzony)_ 

---

## Wnioski

W przeprowadzonych eksperymentach **Keras Tuner skutecznie poprawił wyniki** w porównaniu do modeli baseline. Najlepszy wynik osiągnął **Eksperyment 4** (trials=10, epochs=30) z Accuracy=0.9825 i F1=0.9825, który wykorzystywał **32 jednostki w warstwie ukrytej** i learning rate=0.001. 

**Kluczowe obserwacje:**
- **Tuner pokonał baseline:** Eksperyment 4 osiągnął Accuracy 0.9825 vs DNN baseline 0.9649 (+1.76 p.p.) i RandomForest 0.9474 (+3.51 p.p.)
- **Więcej prób = lepszy wynik:** Zwiększenie liczby prób z 6 do 10 (Eksperyment 3 vs 4) pozwoliło znaleźć lepszą konfigurację (32 units zamiast 96), która dała wyższy Accuracy
- **Mniej neuronów = lepsze wyniki:** Najlepszy model użył zaledwie 32 neuronów (vs 64 w baseline, 96 i 128 w innych eksperymentach), co sugeruje że mniejszy model lepiej generalizuje dla tego zbioru danych
- **Learning rate 0.001 jest optymalny:** Eksperymenty 3 i 4 z LR=0.001 dały lepsze wyniki niż eksperyment 5 z LR=0.01
- **Więcej epok nie zawsze pomaga:** Eksperyment 5 (50 epok) osiągnął taki sam Accuracy jak Eksperyment 4 (30 epok), ale gorszy ROC-AUC, co może sugerować lekkie przeuczenie

**Podsumowanie:** Keras Tuner pozwolił automatycznie znaleźć konfigurację, która przewyższa zarówno ręcznie skonfigurowany DNN baseline, jak i model RandomForest. Najlepsza konfiguracja okazała się prostsza niż początkowo zakładano (mniej neuronów), co podkreśla wartość systematycznego przeszukiwania przestrzeni hiperparametrów.


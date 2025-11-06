## Porównanie technik balansowania i progowania

Celem eksperymentu było porównanie skuteczności różnych technik radzenia sobie z niezbalansowanymi klasami
oraz sprawdzenie, czy zmiana progu decyzyjnego może poprawić wyniki klasyfikacji.

### Wyniki metryk dla modeli

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|--------|-----------|------------|---------|---------|----------|
| Bazowy | 0.94 | 0.00 | 0.00 | 0.00 | 0.81 |
| Ważenie klas | 0.70 | 0.14 | 0.88 | 0.25 | 0.82 |
| SMOTE | 0.71 | 0.14 | 0.88 | 0.25 | **0.82** |
| Undersampling | 0.70 | 0.13 | 0.82 | 0.23 | 0.80 |

### Wykres porównania metryk
Na wykresie porównano Accuracy, Precision, Recall, F1 i ROC AUC dla wszystkich modeli.
Bazowy model osiągnął wysokie Accuracy, ale nie wykrył żadnego przypadku klasy 1 (Recall = 0).
Techniki balansowania (SMOTE, ważenie klas, undersampling) znacząco poprawiły Recall i F1.

### Analiza progu decyzyjnego
Dla modelu bazowego zbadano wpływ zmiany progu klasyfikacji na Precision, Recall i F1-score.
Wraz ze spadkiem progu (np. z 0.5 do 0.3):
- **Recall wzrósł znacząco**, co oznacza, że model wykrywa więcej przypadków klasy 1.
- **Precision spadł**, czyli pojawiło się więcej fałszywych alarmów.
- **Najlepszy kompromis (maksymalny F1)** osiągnięto dla progu ok. **0.3–0.4**.

### Wnioski
- W problemach niezbalansowanych **Accuracy nie odzwierciedla jakości modelu**.
- **SMOTE** i **ważenie klas** dają najlepsze wyniki pod względem Recall i F1-score.
- Regulacja **progu decyzyjnego** może znacząco poprawić wyniki nawet bez zmiany modelu.
- Najlepsze połączenie w tym eksperymencie: **SMOTE + odpowiednio dobrany próg (~0.3)**.

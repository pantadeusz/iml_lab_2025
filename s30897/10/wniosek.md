## WNIOSEK DO LAB 10

### Uwagi:
Program mimo wszelkich prób przerobienia i użycia innych nieco prostszych algorytmów, robił się lokalnie strasznie długo.

### Analiza algorytmów:

| Architektura Modelu | Czas / Epokę (CPU) | Czas / Epokę (GPU) |
| :--- | :--- | :--- |
| **GlobalAveragePooling1D** | ~3 - 8 sek | < 1 sek |
| **GRU (Płytki / Shallow)** | ~45 - 80 sek | ~5 - 8 sek |
| **LSTM (Płytki / Shallow)** | ~60 - 100 sek | ~7 - 10 sek |
| **GRU (Głęboki / Deep)** | ~90 - 150 sek | ~10 - 16 sek |
| **LSTM (Głęboki / Deep)** | ~120 - 200 sek | ~14 - 20 sek |

Dlatego tworząc program, starałem się testować go w Google Colab.

Skrypt predykcyjny poprawnie ładuje zapisany model (`sentiment_model.keras`) i przetwarza surowy tekst wejściowy. Wynik jest skalowany funkcją **Sigmoid** do postaci prawdopodobieństwa sentymentu pozytywnego.

Dla przykładowego tekstu wejściowego:
> *"Totally overhyped. It tries too hard to be funny but falls flat. I wanted my money back."*

Model zwróciłby klasyfikację **NEGATIVE** z wysokim prawdopodobieństwem (np. **95.12% negatywne**), co jest zgodne z rzeczywistym zabarwieniem emocjonalnym recenzji.
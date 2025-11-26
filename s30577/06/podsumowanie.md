## Podsumowanie i Weryfikacja Modelu

### 1. Najlepsze Hiperparametry
W wyniku optymalizacji (Hyperband) najlepsze rezultaty osiągnięto dla konfiguracji:
* **Dokładność (Validation Accuracy):** 54.13%
* **Optymalizator:** `adam` (wykazał znaczącą przewagę nad `sgd`)
* **Funkcja aktywacji:** `elu`
* **Inicjalizator wag:** `he_normal`

### 2. Test na rzeczywistych zdjęciach

**Wyniki predykcji:**
```text
bean3.jpg: bean_rust (49.2%)  [Prawdziwa klasa: leaf_spot] -> BŁĄD
bean2.jpg: bean_rust (49.1%)  [Prawdziwa klasa: rust]      -> POPRAWNIE
bean1.jpg: healthy (75.9%)    [Prawdziwa klasa: healthy]   -> POPRAWNIE
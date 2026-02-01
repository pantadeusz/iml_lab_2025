# 📈 Wniosek z Porównania Autoenkoderów (CAE vs. FAE)

## 🎯 Cel Eksperymentu

Celem eksperymentu była analiza i porównanie wydajności dwóch różnych architektur autoenkoderów na zbiorze danych **Fashion MNIST** z zastosowaniem warstwy **Augmentacji (Obrót)**:

1.  **Autoenkoder Konwolucyjny (CAE):** Wykorzystujący warstwy `Conv2D` i `Conv2DTranspose`.
2.  **Autoenkoder Oparty na Gęstych Warstwach (FAE):** Wykorzystujący jedynie warstwy `Dense` (w pełni połączone).

Oba modele zostały wytrenowane przez **10 epok** z tą samą stratą **Mean Squared Error (MSE)**, aby ocenić ich zdolność do rekonstrukcji obrazów.

## 📊 Analiza Wyników Treningu

| Model | Ostatnia Strata Walidacyjna (val_loss) | Czas Treningu na Epokę | Liczba Parametrów (Szacunkowa) |
| :--- | :--- | :--- | :--- |
| **Autoenkoder Konwolucyjny (CAE)** | **0.0221** | $\approx 13-14 \text{ s}$ | Mniejsza (dzięki udostępnianiu wag) |
| **Autoenkoder Gęsty (FAE)** | **0.0089** | $\approx 2 \text{ s}$ | Większa (Dense(784) ma dużo wag) |

---

### 1. Wydajność Strata (Loss)

* **Zwykły Autoenkoder Gęsty (FAE)** osiągnął znacznie **niższą stratę walidacyjną (0.0089)** w porównaniu do Autoenkodera Konwolucyjnego (0.0221).
* **Wniosek dotyczący straty:** Niższa strata FAE sugeruje, że był on bardziej efektywny w bezpośrednim odwzorowaniu każdego wejściowego piksela na piksel wyjściowy, co jest typowe dla FAE, gdy celem jest **dokładna rekonstrukcja pikseli**. Jednak ta niższa strata nie musi oznaczać lepszej **jakości wizualnej** rekonstrukcji, zwłaszcza w zadaniach redukcji szumów lub uczenia się reprezentacji semantycznej.

### 2. Efektywność Czasowa

* **Autoenkoder Gęsty (FAE)** trenował **znacznie szybciej** ($\approx 2 \text{ s}$ na epokę) niż Autoenkoder Konwolucyjny ($\approx 13 \text{ s}$ na epokę).
* **Wniosek dotyczący czasu:** FAE wymaga znacznie mniej zasobów obliczeniowych na pojedynczą epokę, ponieważ nie wykonuje kosztownych obliczeniowo operacji splotowych.

## 💡 Podsumowanie

| Autoenkoder | Zalety | Wady | Optymalny dla |
| :--- | :--- | :--- | :--- |
| **Konwolucyjny (CAE)** | Uczy się **cech przestrzennych**, generuje **ostrzejsze** rekonstrukcje, lepszy w redukcji szumów (invariance). | Dłuższy czas treningu, wyższa strata MSE w tym teście. | Ekstrakcja cech, Zadania Generatywne, Ograniczenia Danych. |
| **Gęsty (FAE)** | Bardzo **szybki trening**, najniższa strata MSE. | Ignoruje strukturę przestrzenną, rekonstrukcje mogą być **rozmyte** lub mniej semantycznie poprawne. | Bardzo proste zestawy danych, **Szybka kompresja/dekompresja** danych wektorowych. |

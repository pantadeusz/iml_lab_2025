## Wyniki treningu
Trening przeprowadziłem na lokalnej maszynie. Proces był stosunkowo szybki (ok. 1.5 minuty na epokę).

* **Accuracy (zbiór treningowy):** ~83.0%
* **Validation Accuracy (zbiór testowy):** ~82.9%

**Wniosek:** Model uczy się stabilnie i nie wykazuje oznak przeuczenia (overfittingu), ponieważ wyniki na zbiorze walidacyjnym są niemal identyczne jak na treningowym.

## Testy manualne i analiza błędów
Po wytrenowaniu modelu przetestowałem go na własnych przykładach, aby sprawdzić, jak radzi sobie z różnymi typami wypowiedzi.

### Co działa dobrze?
Model świetnie radzi sobie z jednoznacznymi opiniami, gdzie występują silne słowa kluczowe.

> "The acting was incredible and I loved every second of it."
> **Wynik:** POZYTYWNY (Pewność: ~0.55)

> "Boring plot, terrible acting, waste of time. Do not watch."
> **Wynik:** NEGATYWNY (Pewność: -2.40 - bardzo silny wynik!)

### Co sprawia problemy? (Ograniczenia LSTM)
Model gubi się, gdy kontekst zmienia się w trakcie zdania lub gdy recenzja jest sarkastyczna/niejednoznaczna.

**Przypadek 1: Zmiana zdania ("plot twist")**
> "I expected it to be bad, but I was pleasantly surprised. It was actually great!"
> **Wynik modelu:** NEGATYWNY (-0.19)
> **Analiza:** Model prawdopodobnie skupił się na słowie "bad" na początku i nie wyłapał, że spójnik "but" całkowicie odwraca sens wypowiedzi.

**Przypadek 2: Długa recenzja z mieszanymi uczuciami**
> "At first, I thought this movie was going to be another boring drama... [narzekanie] ... However, once the main plot started... [zachwyt] ... Highly recommended!"
> **Wynik modelu:** NEGATYWNY (-0.84)
> **Analiza:** Recenzja zaczynała się od długiego narzekania ("boring", "slow", "unlikable"). Nagromadzenie negatywnych słów na początku przeważyło nad pozytywną końcówką. Model LSTM ma problem z utrzymaniem kontekstu przy tak długich, sprzecznych sekwencjach.

## 5. Podsumowanie
Prosta sieć rekurencyjna osiąga solidną skuteczność (~83%) w ogólnej klasyfikacji. Bardzo dobrze wyłapuje słowa kluczowe ("terrible", "amazing"), ale ma trudności ze zrozumieniem głębszego kontekstu, zaprzeczeń i sarkazmu. Do bardziej zaawansowanej analizy sentymentu (rozumienie kontekstu całego akapitu) lepszym rozwiązaniem byłyby nowocześniejsze architektury oparte na Transformerach (np. BERT).
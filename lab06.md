# Lab 6 - obsługujemy GPU

Na dzisiejszych zajęciach połączymy się z naszymi maszynami obsługującymi GPU oraz postaramy się przygotować mały eksperyment z ciekawszymi danymi. Znowu - jeśli za dużo czasu pracujesz nad jakimś punktem, to znaczy że warto iść dalej/zapytać prowadzącego zajęcia. Nie musisz robić wszystkiego samodzielnie.

## Połączenie z maszyną i konfiguracja

Masz już dane do połączenia się z maszyną na której jest GPU.

Od teraz będziemy na niej działać. Możesz edytować lokalnie, ale należy uruchamiać eksperymenty na zdalnej maszynie.

Proszę pobrać repozytorium z kodem naszych laboratoriów.

Uruchom swój program z poprzednich zajęć. Sprawdź czy działa.

## 1. Zbiór danych

Będziemy dziś testowali trochę więcej parametrów sieci i zobaczymy jak różne algorytmy i inicjalizatory się sprawdzają w prawie praktycznym przykładzie.

Skorzystaj z [Beans](https://www.tensorflow.org/datasets/catalog/beans).

Zastanów się jakiego typu będzie wyjście z sieci neuronowej. Czy to klasyfikacja czy regresja.

## 2. Cel który chcemy osiągnąć

Proszę przygotować eksperyment który postara się zrobić klasyfikator osiągający największą dokładność na wspomnianym zbiorze danych. Tym razem jest on trochę trudniejszy - mamy kolory, a cechy nie są takie oczywiste. Nie martw się jeśli nie będzie dobrze wychodziło. Ten zbiór nie jest łatwy do nauki. Proszę też weryfikować, czy przypadkiem nie przekraczamy pamięci na GPU/CPU. W razie czego - proszę pytać.

PAMIĘTAJ O NOTOWANIU CO ZOSTAŁO ZROBIONE. Może to być nawet w postaci komitów na branchu z odpowiednim komentarzem. Bez notowania postępów będzie trudno odtworzyć co i jak było robione i jakie ścieżki były błędne.

## 3. Sieć neuronowa i jej parametry

Przygotuj sieć neuronową - na razie zwykła warstwową. Sugerują co najmniej 2 warstwy ukryte.

Przygotuj funkcję tworzącą taką sieć, aby można jej było podać różne metody inicjalizacji, różne funkcje aktywacji i różne optymalizatory.

Celowo nie wymieniam. Liczę na indywidualne kombinowanie.

Podpowiedź:

```python
def create_model(initializer='glorot_uniform', activation='relu', optimizer='adam'):
    model = keras.Sequential([
        # ... layers
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

## 4. Optymalizacja hiperparametrów

Sprawdź czy metoda z poprzednich zajęć ciągle jest odpowiednia. Jeśli Keras Tuner nie obsługuje wszystkich parametrów, rozszerz funkcję build_model lub użyj innej metody. Może być nawet metoda siatki.

Nauczony model warto zapisać.

## 5. Finalizowanie

Proszę zrobić dodatkowy program, który będzie po prostu ładował nauczony model i pozwalał na klasyfikację.

## 6. Podsumowanie

Proszę o krótkie podsumowanie - co udało się osiągnąć. Jakie wartości metryk i jakie parametry dały najlepsze rezultaty.

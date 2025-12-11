# Lab 6 – Klasyfikacja zbioru Beans przy użyciu GPU

**Autor:** [Twoje imię]  
**Data:** [Data wykonania laboratorium]  
**Środowisko:** Python 3.x, TensorFlow 2.x, Keras Tuner, GPU

---

## 1. Cel laboratorium

Celem laboratorium było przygotowanie klasyfikatora sieci neuronowej dla zbioru danych **Beans**, optymalizacja hiperparametrów oraz uzyskanie jak największej dokładności klasyfikacji. Laboratorium obejmowało:

- Pracę na zdalnej maszynie z GPU.  
- Stworzenie modelu CNN z kilkoma warstwami konwolucyjnymi i w pełni połączonymi.  
- Testowanie różnych funkcji aktywacji, liczby neuronów, warstw i parametrów treningu.  
- Zapis nauczonego modelu i jego ewaluacja.

---

## 2. Zbiór danych

**Zbiór danych:** Beans (dostępny w `tensorflow_datasets`)  

- Typ zadania: **klasyfikacja wieloklasowa**  
- Liczba klas: 3  
- Dane wejściowe: obrazy RGB o wymiarach 128x128  
- Dane wyjściowe: etykiety klas (sparse_categorical)  

### Przygotowanie danych

- Zmiana rozmiaru obrazu do 128x128  
- Normalizacja pikseli do zakresu [0,1]  
- Shuffle i batching danych (`batch_size=32`)  

---

## 3. Architektura sieci neuronowej

Finalny model zbudowany przy użyciu **Keras Tuner**:  

- **Input:** 128x128x3  
- **Warstwy konwolucyjne:** 3 warstwy Conv2D z 32 filtrami, kernel 3x3, `relu`, MaxPooling2D  
- **Flatten**  
- **Dense:** 192 jednostki, aktywacja `relu`  
- **Dropout:** 0.2  
- **Wyjście:** Dense z liczbą neuronów równą liczbie klas, `softmax`  

**Optymalizator:** Adam  
**Learning rate:** 0.000682  
**Loss:** sparse_categorical_crossentropy  
**Metryka:** accuracy  

---

## 4. Eksperymenty z hiperparametrami

| Próba | Warstwy | Jednostki | Aktywacja | Learning Rate | Dokładność |
|-------|---------|-----------|-----------|---------------|------------|
| 1     | 2       | 384       | selu      | 0.000152      | 0.6250     |
| 2     | 2       | 416       | tanh      | 0.002538      | 0.7188     |
| 3     | 3       | 320       | tanh      | 0.001809      | 0.7578     |
| **Finalny** | 3       | 192       | relu      | 0.000682      | 0.8281     |

**Wnioski z eksperymentów:**  

- Zwiększenie liczby warstw konwolucyjnych poprawiło dokładność.  
- Wybór odpowiedniej funkcji aktywacji miał istotny wpływ (`relu` sprawdził się w finalnym modelu).  
- Dropout = 0.2 pomógł zredukować przeuczenie.  
- Tunowanie liczby neuronów w warstwach Dense pozwoliło znaleźć optymalną konfigurację.  

---

## 5. Trening i ewaluacja

- **EarlyStopping** monitorujący `val_accuracy`, patience = 5  
- Model trenowany do 30 epok, batch_size = 32  
- **Finalny wynik na zbiorze testowym:**  

- Accuracy: 0.8281

- Loss: 0.4982

- Model zapisano w pliku: `beans_model_final.h5`  

---

## 6. Podsumowanie

- Udało się stworzyć model CNN dla zbioru Beans, który osiągnął **dokładność ~82,8%**.  
- Eksperymenty z hiperparametrami pozwoliły znaleźć najlepsze ustawienia: 3 warstwy konwolucyjne, 192 jednostki w Dense, funkcja aktywacji relu, dropout 0.2, learning rate = 0.000682.  
- Zrealizowano wszystkie punkty laboratorium: połączenie z GPU, przygotowanie danych, budowa modelu, optymalizacja hiperparametrów, trening, ewaluacja i zapis modelu.  

**Wnioski:**  

- Większa liczba warstw i neuronów poprawia dokładność, ale wymaga ostrożności ze względu na pamięć GPU.  
- Funkcja aktywacji `relu` i umiarkowany dropout dobrze balansują między nauką a generalizacją.  
- Zapis modelu pozwala na jego późniejsze użycie do klasyfikacji nowych obrazów.

---


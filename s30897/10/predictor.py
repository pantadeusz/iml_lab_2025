import tensorflow as tf
import sys


MODEL_FILENAME = 'sentiment_model.keras'

def load_and_predict():
    try:
        print('Loading model...')
        model = tf.keras.models.load_model(MODEL_FILENAME)
        print('Model loaded.')

    except FileNotFoundError:
        print(f"❌ Błąd: Nie znaleziono pliku modelu **{MODEL_FILENAME}**.")
        print("Upewnij się, że uruchomiłeś **sentiment_trainer.py** i model został zapisany.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Błąd podczas ładowania modelu: {e}")
        sys.exit(1)

    print("\n--- Analiza Sentymentu ---")
    input_text = []
    # 2. Odbiór tekstu z wejścia standardowego
    for line in sys.stdin:
        input_text.append(line.strip())

    if not input_text:
        print("Brak danych wejściowych. Zakończenie programu.")
        return

    full_text = " ".join(input_text)

    # 3. Predykcja
    input_tensor = tf.constant([full_text])
    predictions = model.predict(input_tensor)

    # Przekształcenie logitów na prawdopodobieństwo
    probability = tf.sigmoid(predictions[0])

    # Interpretacja wyniku: logity > 0 lub prawdopodobieństwo > 0.5 to pozytywny
    sentiment = "POZYTYWNY" if probability.numpy()[0] >= 0.5 else "NEGATYWNY"

    # 4. Wyświetlenie predykcji
    print("\n==================================")
    print(f"TEKST WEJŚCIOWY:\n{full_text}")
    print("----------------------------------")
    print(f"PREDYKCJA SENTYMENTU: **{sentiment}**")
    print(f"Prawdopodobieństwo pozytywne: **{probability.numpy()[0] * 100:.2f}%**")
    print("==================================")


if __name__ == "__main__":
    load_and_predict()

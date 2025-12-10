
import sys
import numpy as np
import tensorflow as tf

# Ścieżka do zapisanego modelu
MODEL_PATH = 'sentiment_model.keras'


def load_model(path):
    """Ładuje zapisany model z pliku."""
    print(f"Ładowanie modelu z: {path}", file=sys.stderr)
    try:
        model = tf.keras.models.load_model(path)
        print("Model załadowany pomyślnie!", file=sys.stderr)
        return model
    except Exception as e:
        print(f"Błąd ładowania modelu: {e}", file=sys.stderr)
        sys.exit(1)


def read_input():
    """Czyta tekst ze standardowego wejścia."""
    lines = []
    for line in sys.stdin:
        lines.append(line.strip())
    return ' '.join(lines)


def predict_sentiment(model, text):

    # Model używa from_logits=True, więc output to logit
    prediction = model.predict(
        tf.constant([text], dtype=tf.string),
        verbose=0
    )[0][0]

    # Konwersja logit na prawdopodobieństwo
    probability = tf.sigmoid(prediction).numpy()

    sentiment = 'POZYTYWNY' if prediction > 0 else 'NEGATYWNY'

    return prediction, sentiment, probability


def main():
    """Główna funkcja programu."""
    # Załaduj model
    model = load_model(MODEL_PATH)

    # Przeczytaj tekst ze stdin
    print("Czytam tekst ze standardowego wejścia...", file=sys.stderr)
    text = read_input()

    if not text.strip():
        print("Błąd: Nie podano żadnego tekstu!", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'= ' *50}", file=sys.stderr)
    print("Analizowany tekst:", file=sys.stderr)
    print(text[:200] + ('...' if len(text) > 200 else ''), file=sys.stderr)
    print('= ' *50, file=sys.stderr)

    # Wykonaj predykcję
    prediction, sentiment, probability = predict_sentiment(model, text)

    # Wyświetl wyniki
    print(f"\n{'= ' *50}")
    print("WYNIK ANALIZY SENTYMENTU")
    print('= ' *50)
    print(f"Sentyment: {sentiment}")
    print(f"Pewność: {probability:.2%}")
    print(f"Wartość logit: {prediction:.4f}")
    print('= ' *50)


if __name__ == '__main__':
    main()

import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def main():
    model_path = 'model.keras'
    if not os.path.exists(model_path):
        print(f"Błąd: Nie znaleziono pliku '{model_path}'.")
        return

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Błąd: {e}")
        return

    print("Podaj tekst do analizy (zakończ Ctrl+D lub Ctrl+Z):")
    
    input_text = ""
    for line in sys.stdin:
        input_text += line

    if not input_text.strip():
        return

    prediction = model.predict(tf.constant([input_text]), verbose=0)[0][0]

    if prediction > 0:
        sentiment = "POZYTYWNY"
    else:
        sentiment = "NEGATYWNY"
    
    print("-" * 30)
    print(f"Tekst: {input_text.strip()}")
    print(f"Wynik: {prediction:.4f}")
    print(f"Werdykt: {sentiment}")
    print("-" * 30)

if __name__ == "__main__":
    main()
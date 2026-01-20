import os.path
import sys
from text_classification_rnn import predict_text_sentiment

from keras.models import load_model

def classify_prediction(model, line):
    prediction = predict_text_sentiment(model, line)
    val = prediction[0][0]

    if val > 0.1:
        print("Tekst jest pozytywny")
    elif val < 0.1:
        print("Tekst jest negatywny")
    else:
        print("Tekst jest neutralny")

    print("Wartość predykcji: ", val)


def handle_user_input(model):
    print("Sprawdź sentymentu tekstu wpisując go poniżej. Naciśnij 'Q', aby wyjść z programu")
    for line in sys.stdin:
        if line.strip().lower() == "q":
            print("Zakończono program")
            break

        if not line:
            continue

        classify_prediction(model, line)


if __name__ == '__main__':
    if os.path.exists("sentiment_model.keras"):
        model = load_model("sentiment_model.keras")
    else:
        raise FileNotFoundError("Model file not found")

    handle_user_input(model)
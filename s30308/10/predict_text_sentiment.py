import os.path
import sys
from text_classification_rnn import predict_text_sentiment

from tensorflow.python.keras.saving.save import load_model

if __name__ == '__main__':
    if os.path.exists("sentiment_model.keras"):
        model = load_model("sentiment_model.keras")
    else:
        raise FileNotFoundError("Model file not found")

    print("Sprawdź sentymentu tekstu wpisując go poniżej. Naciśnij 'Q', aby wyjść z programu")
    for line in sys.stdin:
        predict_text_sentiment(model, line)
import tensorflow as tf
import sys
import keras

MODEL_PATH="best_tuner_model.keras"

try:
    model = keras.models.load_model(MODEL_PATH)
except:
    print(f"Błąd ładowania modelu: {MODEL_PATH}")
    sys.exit(1)

text_input = " ".join([line.strip() for line in sys.stdin.readlines()])

if not text_input:
    sys.exit(0)

text_tensor = tf.constant([text_input])
logits = model.predict(text_tensor, verbose=0)
probability = tf.sigmoid(logits[0][0]).numpy()

print(logits)
sentiment = "POZYTYWNY" if probability >= 0.5 else "NEGATYWNY"

print(f"Predykcja: {sentiment}")

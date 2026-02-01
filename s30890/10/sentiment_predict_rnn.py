import sys
import tensorflow as tf


MODEL_DIR = "sentiment_rnn_model.keras"


def load_model(model_dir=MODEL_DIR):
    model = tf.keras.models.load_model(model_dir)
    return model


def read_text_from_stdin():
    lines = [line for line in sys.stdin]
    text = "".join(lines).strip()
    return text


def predict_sentiment(model, text, threshold=0.0):
    if not text:
        raise ValueError("Brak tekstu do klasyfikacji – standardowe wejście było puste.")
    inputs = tf.constant([text], dtype=tf.string)
    logits = model(inputs)
    logit_value = float(logits[0][0].numpy())
    probability_positive = float(tf.sigmoid(logit_value).numpy())
    label_str = "pozytywny" if logit_value >= threshold else "negatywny"

    return label_str, logit_value, probability_positive


def main():
    model = load_model(MODEL_DIR)
    text = read_text_from_stdin()
    label, logit_value, prob_pos = predict_sentiment(model, text)

    print(f"Predykcja: {label}")
    print(f"logit = {logit_value:.4f}")
    print(f"komentarz pozytywny = {prob_pos:.4f}")


if __name__ == "__main__":
    main()

import tensorflow as tf
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_FILENAME = 'model_sentymentu.keras'


def load_model(path):
    if not os.path.exists(path):
        print(f"Brak pliku modelu: {path}")
        sys.exit(1)
    return tf.keras.models.load_model(path)


def make_prediction(model, text):
    input_data = tf.constant([text])
    predictions = model.predict(input_data, verbose=0)
    return predictions[0][0]


def main():
    model = load_model(MODEL_FILENAME)

    print("Podaj tekst(Ctrl+D):")

    input_lines = []
    try:
        for line in sys.stdin:
            input_lines.append(line)
    except KeyboardInterrupt:
        pass

    full_text = "".join(input_lines).strip()

    if not full_text:
        return

    score = make_prediction(model, full_text)

    print("-" * 30)
    if score > 0:
        print(f"POZYTYWNY ({score:.4f})")
    else:
        print(f"NEGATYWNY ({score:.4f})")
    print("-" * 30)


if __name__ == "__main__":
    main()
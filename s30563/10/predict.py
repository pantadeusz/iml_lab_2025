import sys
from sys import argv
import os
import numpy as np
import tensorflow as tf

def predict_line(model, text):
    input_data = tf.constant([text.strip()], dtype=tf.string)

    predictions = model.predict(input_data, verbose=0)
    score = predictions[0][0]

    return tf.sigmoid(score).numpy()


def main():
    model_path = argv[1]
    model = tf.keras.models.load_model(model_path)

    print("Podaj opinię")
    for line in sys.stdin:
        result = predict_line(model, line)

        print("Pozytywna" if result > 0.5 else "Negatywna")
        print("Podaj kolejną opinię (ctrl + c, aby wyjść)")


if __name__ == "__main__":
    if len(argv) > 1 and os.path.exists(argv[1]) and argv[1].endswith(".keras"):
        main()
    else:
        print("ERROR: Podaj poprawną ścieżkę do modelu jako argument.")
        print("python predict.py model_path")
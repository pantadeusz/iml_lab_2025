import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    model = load_model("model.keras")

    print("Enter comment, or \"q\" to exit:")

    for line in sys.stdin:
        if line.strip() == "q":
            exit(0)

        predictions = model.predict(tf.constant([line], dtype=tf.string), verbose=0)

        if predictions[0] >= 0:
            print("Positive comment")
        else:
            print("Negative comment")

        print("Enter comment, or \"q\" to exit:")


if __name__ == "__main__":
    main()

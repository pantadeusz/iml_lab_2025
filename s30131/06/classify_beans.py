import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds


def load_labels(path: str):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/beans_best.keras")
    parser.add_argument("--labels", type=str, default="models/labels.txt")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    labels = load_labels(args.labels)

    if args.img_size is None:
        h, w = model.input_shape[1:3]
    else:
        h = w = args.img_size

    if args.image:
        img = keras.utils.load_img(args.image, target_size=(h, w))
        x = keras.utils.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        y_true = None
    else:
        ds = tfds.load("beans", split="test", as_supervised=True)
        img, y_true = next(iter(ds.shuffle(1000).take(1)))
        img = tf.image.resize(img, (h, w))
        x = (tf.cast(img, tf.float32) / 255.0)[None, ...].numpy()

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    print(f"Predykcja: {labels[idx]} (p={probs[idx]:.3f})")
    print("Rozkład:", {labels[i]: float(f"{probs[i]:.3f}") for i in range(len(labels))})
    if 'y_true' in locals() and y_true is not None:
        print(f"Prawda: {labels[int(y_true)]}")


if __name__ == "__main__":
    main()



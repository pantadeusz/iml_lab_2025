import tensorflow as tf
import sys
import numpy as np

MODEL_FILENAME = 'sentiment_model.keras'


def load_and_predict():
    try:
        model = tf.keras.models.load_model(MODEL_FILENAME)
        print('Model loaded successfully.')

    except FileNotFoundError:
        print(f"Failure! The model file **{MODEL_FILENAME}** was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Failure while loading model: {e}")
        sys.exit(1)

    encoder_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.TextVectorization):
            encoder_layer = layer
            break

    vocabulary = np.array(encoder_layer.get_vocabulary())

    print(f"Directory contains {len(vocabulary):,} tokens.")
    print(f"First 20 of them: {vocabulary[:20]}")
    # -----------------------------------------------------------

    print("\n--- Analyzing sentiment ---")

    input_text = ["Totally overhyped. It tries too hard to be funny but falls flat. I wanted my money back."]
    for line in sys.stdin:
        input_text.append(line.strip())

    if not input_text:
        print("No input data provided.")
        return

    full_text = " ".join(input_text)

    input_tensor = tf.constant([full_text])
    predictions = model.predict(input_tensor, verbose=0)

    probability = tf.sigmoid(predictions[0])

    positive_prob_value = probability.numpy()[0]
    sentiment = "POSITIVE" if positive_prob_value >= 0.5 else "NEGATIVE"

    print("\n==================================")
    print(f"INPUT TEXT:\n{full_text}")
    print("----------------------------------")
    print(f"SENTIMENT PREDICTION: {sentiment}")
    print(f"Positive probability: {positive_prob_value * 100:.2f}%")
    print(f"Negative probability: {100 - (positive_prob_value * 100):.2f}%")
    print("==================================")


if __name__ == "__main__":
    load_and_predict()
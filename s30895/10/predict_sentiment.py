import keras.models
import sys
import tensorflow as tf


def predict(model,text):
    sample_text = [text]
    sample_tensor = tf.constant(sample_text)

    predictions = model.predict(sample_tensor)
    probabilities = tf.sigmoid(predictions).numpy()
    predicted_class = (probabilities > 0.5).astype(int)

    if predicted_class[0] == 1:
        print("Predicted class: Positive")
    else:
        print("Predicted class: Negative")


try:
    model = keras.models.load_model("model.keras")
except Exception:
    print("couldn't load the model")
    exit(1)

for line in sys.stdin:
    predict(model,line)

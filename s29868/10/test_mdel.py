import sys
import tensorflow as tf

model = tf.keras.models.load_model('rnn_model.keras')

text = sys.stdin.read().strip()
score = model.predict(tf.constant([text]), verbose=0)[0][0]

print(f"\n\n\n\n\nText put in model: {text}")
print("POSITIVE" if score >= 0 else "NEGATIVE")
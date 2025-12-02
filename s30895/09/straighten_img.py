import sys
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

if sys.argv.__len__() < 2:
    print("source image argument needed")

img_path = sys.argv[1]

img = Image.open(img_path).convert("L")
img_np = np.array(img).astype("float32") / 255.0
img_np = np.expand_dims(img_np, axis=(0, -1))  # (1, 28, 28, 1)

encoder = tf.keras.models.load_model("models/encoder.keras")
decoder = tf.keras.models.load_model("models/decoder.keras")

latent = encoder.predict(img_np)
straightened = decoder.predict(latent)

output_img = straightened[0, :, :]
plt.imsave("straightened_img.png", output_img, cmap="gray")

print("Saved output image: fashion_image_after_encoding.png")

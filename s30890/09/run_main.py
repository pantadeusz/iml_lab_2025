import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

#wczytanie
encoder = keras.models.load_model("encoder_model.h5")
decoder = keras.models.load_model("decoder_model.h5")


def load_image(path):
    img = Image.open(path).convert("L").resize((28, 28))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

input_img = load_image("input.png")


latent = encoder.predict(input_img)
output_img = decoder.predict(latent)[0]

print("Wektor latent (ukryty):")
print(latent)

#zapis
output_img = (output_img * 255).astype(np.uint8)
Image.fromarray(output_img.squeeze()).save("output.png")

print("Zapisano wynik jako output.png")

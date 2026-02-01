import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import fashion_mnist

encoder_path = 'encoder.keras'
decoder_path = 'decoder.keras'

image_path = ""

def load_image(img_path):
    img = image.load_img(img_path, target_size=(28,28), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = img_array.astype('float32') / 255
    input_image = np.expand_dims(img_array, axis=0)
    return input_image

try:
    loaded_encoder = load_model(encoder_path, compile=False)
    loaded_decoder = load_model(decoder_path, compile=False)
    print("Loaded encoder and decoder models")
except Exception as e:
    print(f"\n❌ Nie udało się załadować obrazka. Błąd: {e}")
    exit()


print(f"\nŁadowanie obrazu z: {image_path}")
try:
    input_image = load_image(image_path)
    print(f"Kształt załadowanego obrazka (batch, h, w, kanały): {input_image.shape}")
except Exception as e:
    print(f"\n❌ Nie udało się załadować obrazka. Błąd: {e}")
    (_, _), (x_test, _) = fashion_mnist.load_data()
    image_index = np.random.randint(0, len(x_test))
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255.
    input_image = x_test[image_index:image_index + 1]


rotation_layer = layers.RandomRotation(factor=0.5)
input_image_rotated = rotation_layer(input_image, training=True).numpy()

latent_vector = loaded_encoder.predict(input_image_rotated)
reconstructed_image = loaded_decoder.predict(latent_vector)

print("\n--- Wektor Ukryty (Latent) ---")
print(latent_vector[0])
print(f"\nKształt wektora ukrytego: {latent_vector.shape}")

output_filename = f'rekonstrukcja_zewnetrzny.png'
try:
    plt.imsave(output_filename, reconstructed_image[0].squeeze(), cmap='gray')
    print(f"✅ Zrekonstruowany obraz zapisano jako: {output_filename}")
except Exception as e:
    print(f"❌ Błąd podczas zapisu pliku {output_filename}: {e}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle(f"Autoenkoder Denoising/Robust dla obrazka zewnętrznego")
axes[0].imshow(input_image[0].squeeze(), cmap='gray')
axes[0].set_title("Oryginalny (Cel)")
axes[0].axis('off')

axes[1].imshow(input_image_rotated[0].squeeze(), cmap='gray')
axes[1].set_title("Obrócony (Wejście)")
axes[1].axis('off')

axes[2].imshow(reconstructed_image[0].squeeze(), cmap='gray')
axes[2].set_title("Zrekonstruowany")
axes[2].axis('off')

plt.show()
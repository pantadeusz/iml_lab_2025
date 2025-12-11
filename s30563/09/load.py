import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array


def main():
    _, model_folder_path, img_path = sys.argv

    if not os.path.exists(model_folder_path):
        print(
            f"Nie znalezniono folderu modelu (encoder + decoder) w {model_folder_path}"
        )
    elif not os.path.exists(img_path):
        print(f"Nie znaleziono zdjęcia w {img_path}")

    encoder_path = os.path.join(model_folder_path, "encoder.keras")
    decoder_path = os.path.join(model_folder_path, "decoder.keras")

    encoder = load_model(encoder_path)
    decoder = load_model(decoder_path)

    img = load_img(img_path, color_mode="grayscale", target_size=(28, 28))
    img_array = img_to_array(img) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    latent_vector = encoder.predict(input_tensor)
    output_img = decoder.predict(latent_vector)

    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(latent_vector[0])

    result_img = np.squeeze(output_img)

    save_path = "load_result.png"
    plt.imsave(save_path, result_img, cmap="gray")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main()
    else:
        print("Za mało argumentów. Poprawna konstrukcja:")
        print(
            "python load.py <model_path> (folder with encoder and decoder named encoder.keras and decoder.keras) <path_to_img>(28 x 28)"
        )

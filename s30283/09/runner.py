import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

IMAGE_SHAPE = (28, 28)
LATENT_DIM = 64 

class Autoencoder(Model):
    def __init__(self, latent_dim=LATENT_DIM, shape=IMAGE_SHAPE):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=shape + (1,)), 
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'), 
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def init_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--encoder_path', default='encoder.keras')
    parser.add_argument('-d', '--decoder_path', default='decoder.keras') 
    parser.add_argument('-i', '--image_path', default='image.png')
    return parser.parse_args()

def run_inference(encoder, decoder, input_data):
    encoded_img = encoder(input_data)
    reconstructed_img = decoder(encoded_img)
    return reconstructed_img

def load_and_preprocess_image(image_path, target_size=IMAGE_SHAPE):
    img = load_img(image_path, target_size=target_size, color_mode='grayscale')
    np_image = img_to_array(img) / 255.0
    
    return np.expand_dims(np_image, axis=0)


if __name__ == '__main__':
    args = init_argparser()
    encoder_path, decoder_path, image_path = args.encoder_path, args.decoder_path, args.image_path
    
    CUSTOM_OBJECTS = {"Autoencoder": Autoencoder}
    
    try:
        encoder = load_model(encoder_path, custom_objects=CUSTOM_OBJECTS)
        decoder = load_model(decoder_path, custom_objects=CUSTOM_OBJECTS)
    except Exception as e:
        print(f"Error loading models: {e}.")
        exit()
        
    np_image_batch = load_and_preprocess_image(image_path)
    np_image_original = np_image_batch.squeeze()
    
    image_result_batch = run_inference(encoder, decoder, np_image_batch)
    
    image_result = image_result_batch.numpy().squeeze()
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    axes[0].imshow(np_image_original, cmap='gray')
    axes[0].set_title('Original')
    
    axes[1].imshow(image_result, cmap='gray')
    axes[1].set_title('After inference')
    
    plt.tight_layout()
    plt.show()
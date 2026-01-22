import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
from keras import layers

data_augmentation = layers.RandomRotation(
    factor=0.2, 
    fill_mode='constant', 
    fill_value=0.0
)

def preprocess_image(image_path=None, from_dataset=False):
    if from_dataset:
        from keras.datasets import fashion_mnist
        (_, _), (x_test, _) = fashion_mnist.load_data()
        
        x_test = x_test.astype('float32') / 255.
        x_test = np.expand_dims(x_test, -1)

        idx = np.random.randint(0, len(x_test))
        img = x_test[idx]
        
        img_tensor = tf.convert_to_tensor(img)
        img_tensor = tf.expand_dims(img_tensor, 0)

        img_rotated = data_augmentation(img_tensor, training=True)

        return img_rotated.numpy().squeeze(axis=0)
    
    else:
        # Load local image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
        # Resize to match model input
        img = cv2.resize(img, (28, 28))
        img = img.astype('float32') / 255.
        img = np.expand_dims(img, -1)
        return img

def process_single_image(image_input):
    # Load Models
    print("Loading models...")
    try:
        encoder = load_model('encoder_model.keras')
        decoder = load_model('decoder_model.keras')
    except OSError:
        print("Error: Model files not found. Please run the training script first.")
        return

    input_batch = np.expand_dims(image_input, axis=0)

    latent_vector = encoder.predict(input_batch, verbose=0)
    reconstructed_batch = decoder.predict(latent_vector, verbose=0)
    
    output_image = reconstructed_batch[0] 

    # Print Latent Vector
    print("\n" + "="*30)
    print(f"Latent Vector Shape: {latent_vector.shape}")
    print("Latent Vector Values:")
    print(latent_vector)
    print("="*30 + "\n")

    # Save and Visualize
    plt.figure(figsize=(8, 4))
    
    # Input
    ax = plt.subplot(1, 2, 1)
    plt.imshow(image_input.squeeze(), cmap='gray')
    plt.title("Input Image")
    plt.axis('off')

    # Output
    ax = plt.subplot(1, 2, 2)
    plt.imshow(output_image.squeeze(), cmap='gray')
    plt.title("Reconstructed Image")
    plt.axis('off')

    save_path = "inference_result.png"
    plt.savefig(save_path)
    print(f"Result saved to: {save_path}")
    plt.savefig("inference_result.png")

if __name__ == "__main__":
    # print("Testing with random dataset image...")
    # test_img = preprocess_image(from_dataset=True)
    # process_single_image(test_img)

    img_path = "image_proxy.jpg"
    test_img = preprocess_image(image_path=img_path)
    process_single_image(test_img)

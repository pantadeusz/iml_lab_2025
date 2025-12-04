from matplotlib import pyplot as plt
from base_model import load_mnist_data
from model_evaluation import augment_func
def check_augmentation():

    ds_train, _, _ = load_mnist_data(batch_size=8, augment=False)

    images, labels = next(iter(ds_train))

    aug_images, _ = augment_func(images, labels)


    plt.figure(figsize=(10, 4))
    for i in range(8):
        plt.subplot(2, 8, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title("Orig")
        plt.axis('off')

        plt.subplot(2, 8, i + 9)
        plt.imshow(aug_images[i].numpy().squeeze(), cmap='gray')
        plt.title("Aug")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    check_augmentation()

if __name__ == '__main__':
    main()
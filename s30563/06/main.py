import tensorflow_datasets as tfds

def get_dataset():
    train_ds, val_ds, test_ds = tfds.load(
        'beans',
        split=['train', 'validation', 'test'],
        as_supervised=True
    )
    return train_ds, val_ds, test_ds

def main():
    pass

if __name__ == "__main__":
    main()


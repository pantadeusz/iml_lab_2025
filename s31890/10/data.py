import tensorflow as tf
from tensorflow._api.v2.data import AUTOTUNE
import tensorflow_datasets as tfds
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import keras
import numpy as np


def get_imdb(split=["train", "test"]):
    datasets, info = tfds.load("imdb_reviews", split=split, as_supervised=True, with_info=True)

    train_ds, test_ds = datasets

    assert isinstance(train_ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)

    return train_ds, test_ds, info


def print_info(info):
    print("Label names:", info.features["label"].names)
    print("Train size:", info.splits["train"].num_examples)
    print("Test size:", info.splits["test"].num_examples)


def print_sample_info(sample):
    image, label = sample
    print(f"Sample image shape: {image.shape}")
    print(f"Sample image dtype: {image.dtype}")
    print(f"Sample label type: {label.dtype}")
    print(f"Sample label value: {label.numpy()}")


def get_sample_shape(sample):
    image, label = sample
    return image.shape[1:]


def compute_class_weights(dataset):
    labels = []

    for _, y in dataset:
        y_np = y.numpy()

        # Flatten to handle batches or scalar labels
        y_np = np.array(y_np).reshape(-1)

        # If labels are one-hot encoded → convert to class index
        if y_np.ndim > 1 and y_np.shape[-1] > 1:
            y_np = np.argmax(y_np, axis=-1)

        labels.extend(y_np.tolist())

    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(counts)

    weights = {cls: total / (num_classes * count) for cls, count in counts.items()}
    return counts, weights


def prepare(ds, info, batch_size=8, shuffle=False):
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(info.splits["train"].num_examples)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds.prefetch(buffer_size=AUTOTUNE)


def save_tuner_summary(tuner, csv_filename="tuner_trials_summary.csv", num_trials=None):
    if num_trials is None:
        num_trials = len(tuner.oracle.trials)

    all_trials = tuner.oracle.get_best_trials(num_trials=num_trials)

    # Save CSV summary
    trial_data = []
    for trial in all_trials:
        trial_data.append(
            {
                "Trial ID": trial.trial_id,
                "Score": trial.score,
                "Hyperparameters": trial.hyperparameters.values,
            }
        )

    df = pd.DataFrame(trial_data)
    df.to_csv(csv_filename, index=False)

    print(f"Detailed trial summary saved to '{csv_filename}'")
    print(f"Total number of trials analyzed: {len(all_trials)}")

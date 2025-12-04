import tensorflow as tf
import tensorflow_datasets as tfds
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd


def get_beans(split=["train", "validation", "test"]):
    datasets, info = tfds.load(
        "beans",
        split=split,
        as_supervised=True,
        with_info=True
    )
    
    train_ds, val_ds, test_ds = datasets

    assert isinstance(train_ds, tf.data.Dataset)
    assert isinstance(val_ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)
    
    return train_ds, val_ds, test_ds, info

def print_info(info):
    print("Label names:", info.features["label"].names)
    print("Train size:", info.splits["train"].num_examples)
    print("Validation size:", info.splits["validation"].num_examples)
    print("Test size:", info.splits["test"].num_examples)

def print_sample_info(sample):
    image, label = sample
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Label type: {label.dtype}")
    print(f"Label value: {label.numpy()}")

def get_sample_shape(sample):
    image, label = sample
    return image.shape

def compute_class_weights(dataset):
    labels = []
    for _, label in dataset:
        labels.append(int(label.numpy()))
    counts = Counter(labels)
    total = sum(counts.values())
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    return counts, weights

def save_tuner_summary(
    tuner,
    csv_filename="tuner_trials_summary.csv",
    num_trials=None
):
    if num_trials is None:
        num_trials = len(tuner.oracle.trials)

    all_trials = tuner.oracle.get_best_trials(num_trials=num_trials)

    # Save CSV summary
    trial_data = []
    for trial in all_trials:
        trial_data.append({
            'Trial ID': trial.trial_id,
            'Score': trial.score,
            'Hyperparameters': trial.hyperparameters.values
        })

    df = pd.DataFrame(trial_data)
    df.to_csv(csv_filename, index=False)

    print(f"Detailed trial summary saved to '{csv_filename}'")
    print(f"Total number of trials analyzed: {len(all_trials)}")


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

def plot_tuner_results(tuner, plot_filename="tuner_iterations_plot.png", csv_filename="tuner_trials_summary.csv", num_trials=None):
    all_trials = tuner.oracle.get_best_trials(num_trials=num_trials)
    
    trial_numbers = [trial.number for trial in all_trials]
    objective_values = [trial.score for trial in all_trials]
    
    plt.figure(figsize=(12, 6))
    plt.plot(trial_numbers, objective_values, marker='o', linestyle='-', color='b')
    plt.title(f'Keras Tuner: Trial Performance Over Time (Total Trials: {len(all_trials)})')
    plt.xlabel('Trial Number')
    plt.ylabel('Validation Accuracy')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(plot_filename)
    plt.close()
    
    trial_data = []
    for trial in all_trials:
        trial_data.append({
            'Trial Number': trial.number,
            'Score': trial.score,
            'Hyperparameters': trial.hyperparameters.values
        })
    
    df = pd.DataFrame(trial_data)
    df.to_csv(csv_filename, index=False)
    
    print(f"✅ Detailed trial summary saved to '{csv_filename}'")
    print(f"✅ Performance plot saved to '{plot_filename}'")
    print(f"📊 Total number of trials analyzed: {len(all_trials)}")


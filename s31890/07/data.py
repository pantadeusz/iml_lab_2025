import tensorflow as tf
import tensorflow_datasets as tfds
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# ---- Info wrapper (mimics tfds.core.DatasetInfo) ----
class SimpleSplit:
    def __init__(self, num_examples):
        self.num_examples = num_examples

class SimpleLabelFeature:
    def __init__(self, names):
        self.names = names

class SimpleFeatures:
    def __init__(self, label_names, feature_names):
        self.label = SimpleLabelFeature(label_names)
        self.feature_names = feature_names

class SimpleInfo:
    def __init__(self, feature_names, label_names, splits):
        self.features = SimpleFeatures(label_names, feature_names)
        self.splits = {k: SimpleSplit(v) for k, v in splits.items()}


def get_wine(split=["train", "validation", "test"], simple=False):
    wine = fetch_ucirepo(id=109)

    # Extract features and targets
    X = wine.data.features
    y = wine.data.targets

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame with correct column names
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    df = pd.concat([X_scaled, y], axis=1)
        
    df["class"] = df["class"] - 1

    # Split data
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['class']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['class']
    )

    if simple:
        # Extract features and labels
        X_train, y_train = train_df.drop('class', axis=1).values, train_df['class'].values
        X_val, y_val = val_df.drop('class', axis=1).values, val_df['class'].values
        X_test, y_test = test_df.drop('class', axis=1).values, test_df['class'].values

        # Return the split datasets
        return X_train, X_val, X_test, y_train, y_val, y_test

    def to_dataset(df, shuffle=True):
        features = df.drop('class', axis=1).astype('float32').values
        labels = df['class'].astype('int32').values
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            ds = ds.shuffle(len(df))
        return ds

    train_ds = to_dataset(train_df, shuffle=True)
    val_ds   = to_dataset(val_df, shuffle=False)
    test_ds  = to_dataset(test_df, shuffle=False)

    # Fake TFDS-like Info object
    info = SimpleInfo(
        feature_names=list(X.columns),
        label_names=sorted(df["class"].unique()),
        splits={
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        }
    )

    return train_ds, val_ds, test_ds, info

def print_info(info):
    print("Label names:", info.features.label.names)
    print("Train size:", info.splits["train"].num_examples)
    print("Validation size:", info.splits["validation"].num_examples)
    print("Test size:", info.splits["test"].num_examples)


def print_sample_info(sample):
    features, label = sample

    print("Feature tensor shape:", features.shape)
    print("Feature dtype:", features.dtype)
    print("Label dtype:", label.dtype)
    print("Label value:", int(label.numpy()))

def get_sample_shape(sample):
    features, _ = sample
    return features.shape

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


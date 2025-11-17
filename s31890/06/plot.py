import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast


def plot_score_vs_learning_rate(trials_df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=trials_df,
        x="learning_rate",
        y="score",
        hue="optimizer",
        style="use_batch_norm",
        s=100,
        alpha=0.8,
    )
    plt.title("Score vs. Learning Rate (Colored by Optimizer & Batch Norm)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Score")
    plt.xscale("log")  # Log scale for learning rate
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("plot_score_vs_learning_rate.png")


def plot_score_by_dense_layers(trials_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=trials_df,
        x="num_dense_layers",
        y="score",
        estimator="mean",
        ci="sd",
        palette="viridis",
    )
    plt.title("Average Score by Number of Dense Layers")
    plt.xlabel("Number of Dense Layers")
    plt.ylabel("Average Score")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("plot_score_by_dense_layers.png")


def visualize_all_trials(trials_data):
    if isinstance(trials_data, list):
        trials_df = pd.DataFrame(trials_data)
    else:
        trials_df = trials_data.copy()

    # Ensure 'score' is numeric
    trials_df["score"] = pd.to_numeric(trials_df["score"], errors="coerce")

    # Drop rows with missing scores
    trials_df = trials_df.dropna(subset=["score"])

    print("Running: Scatter Plot (Score vs Learning Rate)...")
    plot_score_vs_learning_rate(trials_df)

    print("Running: Bar Chart (Score by Dense Layers)...")
    plot_score_by_dense_layers(trials_df)

    print("All visualizations completed.")


trials_df = pd.read_csv("tuner_trials_summary.csv")
assert isinstance(trials_df, pd.DataFrame)

trials_df.columns = trials_df.columns.str.lower().str.replace(" ", "_")

trials_df["hyperparameters"] = trials_df["hyperparameters"].apply(ast.literal_eval)
hyperparam_df = pd.json_normalize(trials_df["hyperparameters"])
trials_cleaned = pd.concat([trials_df[["trial_id", "score"]], hyperparam_df], axis=1)
# Ensure numeric
trials_cleaned["score"] = pd.to_numeric(trials_cleaned["score"], errors="coerce")
# Drop broken or empty rows
trials_cleaned = trials_cleaned.dropna(subset=["score"])

visualize_all_trials(trials_cleaned)

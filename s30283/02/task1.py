from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import numpy as np
import argparse


def get_threshold():
    parser = argparse.ArgumentParser(
        description="Test how how model probability threshold affects sensitivity and specificity"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold: [0, 1], default=0.5",
    )
    args = parser.parse_args()
    return args.threshold


def prepare_data():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def get_trained_model(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=3000)
    model.fit(X_train, y_train)
    return model


def manual_confusion_matrix(y_test, y_pred):
    # 1 [[TP, FN]]
    # 0 [[FP, TN]]
    #      1   0
    TP = np.sum((y_test == y_pred)[y_test == 1])
    TN = np.sum((y_test == y_pred)[y_test == 0])
    FP = np.sum((y_test != y_pred)[y_test == 0])
    FN = np.sum((y_test != y_pred)[y_test == 1])
    return np.array([[TP, FN], [FP, TN]])


def create_and_save_cm_plot(cm):
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig("confusion_matrix.png")
    plt.close()


def manual_classification_report(y_true, y_pred, output_dict=True):
    confusion_matrix = manual_confusion_matrix(y_true, y_pred)
    TP = confusion_matrix[0, 0]
    FN = confusion_matrix[0, 1]
    FP = confusion_matrix[1, 0]
    TN = confusion_matrix[1, 1]
    # 1 [[TP, FN]]
    # 0 [[FP, TN]]
    #      1   0

    precision1 = float(TP / (TP + FP))
    recall1 = float(TP / (TP + FN))
    f1score1 = float(2 * (precision1 * recall1) / (precision1 + recall1))
    support1 = float(TP + FN)
    precision2 = float(TN / (TN + FN))
    recall2 = float(TN / (TN + FP))
    f1score2 = float(2 * (precision2 * recall2) / (precision2 + recall2))
    support2 = float(TN + FP)
    accuracy = float((TP + TN) / (TP + TN + FP + FN))
    stats = {
        "0": {
            "precision": precision2,
            "recall": recall2,
            "f1-score": f1score2,
            "support": support2,
        },
        "1": {
            "precision": precision1,
            "recall": recall1,
            "f1-score": f1score1,
            "support": support1,
        },
        "accuracy": accuracy,
        "macro_avg": {
            "precision": (precision1 + precision2) / 2,
            "recall": (recall1 + recall2) / 2,
            "f1-score": (f1score1 + f1score2) / 2,
            "support": support1 + support2,
        },
        "weighted_avg": {
            "precision": (precision1 * support1 + precision2 * support2)
            / (support1 + support2),
            "recall": (recall1 * support1 + recall2 * support2) / (support1 + support2),
            "f1-score": (f1score1 * support1 + f1score2 * support2)
            / (support1 + support2),
            "support": support1 + support2,
        },
    }
    if output_dict:
        return stats

    result = ""
    # :> alignment (spacing)
    # .(number)f float formatting
    result += (
        f"{'':<13}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}\n"
    )
    for label in ["0", "1"]:
        row = stats[label]
        result += f"{label:<13}{row['precision']:>10.2f}{row['recall']:>10.2f}{row['f1-score']:>10.2f}{int(row['support']):>10}\n"
    result += f"\n{'accuracy':<33}{stats['accuracy']:>10.2f}{int(stats['macro_avg']['support']):>10}\n"
    for avg in ["macro_avg", "weighted_avg"]:
        row = stats[avg]
        result += f"{avg.replace('_', ' '):<13}{row['precision']:>10.2f}{row['recall']:>10.2f}{row['f1-score']:>10.2f}{int(row['support']):>10}\n"
    return result


def run_proba_experiment(model, X_test, threshold: float = 0.5):
    preds_proba = model.predict_proba(X_test)
    first_class_preds_proba = preds_proba[:, 1]
    y_pred_custom = (first_class_preds_proba > threshold).astype(int)
    tp, fn, fp, tn = manual_confusion_matrix(y_test, y_pred_custom).flatten()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")

    # Own experiment
    # history = {'sensitivity': [], 'specificity': []}
    # thresholds = np.arange(0.01, 0.99, 0.01)

    # for threshold in thresholds:
    #     y_pred_custom = (first_class_preds_proba > threshold).astype(int)
    #     tp, fn, fp, tn = manual_confusion_matrix(y_test, y_pred_custom).flatten()

    #     sensitivity = tp / (tp + fn)
    #     specificity = tn / (tn + fp)

    #     history['sensitivity'].append(sensitivity)
    #     history['specificity'].append(specificity)

    # plt.figure(figsize=(9, 5))
    # plt.plot(thresholds, history['sensitivity'], label='Sensitivity (Recall)', linewidth=2)
    # plt.plot(thresholds, history['specificity'], label='Specificity', linewidth=2, linestyle='--', color='red')
    # plt.title('Effect of Decision Threshold on Sensitivity and Specificity', fontsize=13)
    # plt.xlabel('Decision Threshold', fontsize=12)
    # plt.ylabel('Metric Value', fontsize=12)
    # plt.legend()
    # plt.grid(alpha=0.3)
    # plt.ylim(0, 1.05)
    # plt.savefig('threshold_effects.png')
    # plt.show()


if __name__ == "__main__":
    # Model preparation
    X_train, X_test, y_train, y_test = prepare_data()
    model = get_trained_model(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion matrices
    cm = confusion_matrix(y_test, y_pred)
    manual_cm = manual_confusion_matrix(y_test, y_pred)
    print(f"Sklearn cm:\n{cm}")
    print(f"Manual cm:\n{manual_cm}")
    create_and_save_cm_plot(cm)

    # Classification reports
    print(classification_report(y_test, y_pred, output_dict=False))
    print(f"\nManual:\n{manual_classification_report(y_test, y_pred)}")
    print(
        f"\nManual:\n{manual_classification_report(y_test, y_pred, output_dict=False)}"
    )

    # Threshold experiment
    threshold = get_threshold()
    run_proba_experiment(model, X_test, threshold)

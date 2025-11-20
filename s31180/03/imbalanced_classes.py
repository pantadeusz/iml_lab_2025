from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np


def generate_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_models(X_train, y_train):
    X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
    X_under, y_under = RandomUnderSampler(random_state=42).fit_resample(
        X_train, y_train
    )

    models = {
        "Base": LogisticRegression(random_state=42, max_iter=1000).fit(
            X_train, y_train
        ),
        "Weighted": LogisticRegression(
            class_weight="balanced", random_state=42, max_iter=1000
        ).fit(X_train, y_train),
        "SMOTE": LogisticRegression(random_state=42, max_iter=1000).fit(
            X_smote, y_smote
        ),
        "Under": LogisticRegression(random_state=42, max_iter=1000).fit(
            X_under, y_under
        ),
    }
    return models


def evaluate_models(models, X_test, y_test):
    print("OCENA MODELI (threshold = 0.5)")

    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)

        print(f"\n{name}")
        print(
            classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"])
        )

        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        results.append(
            {
                "Method": name,
                "Precision": report["1"]["precision"],
                "Recall": report["1"]["recall"],
                "F1": report["1"]["f1-score"],
            }
        )

    return results


def threshold_analysis(models, X_test, y_test):
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.8]

    print("ANALIZA PROGOWANIA")

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f"\nModel: {name}")
        print(
            f"{'Threshold':<10} {'Pos%':<8} {'Precision':<10} {'Recall':<10} {'F1':<10}"
        )

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)

            report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )

            prec = report["1"]["precision"]
            rec = report["1"]["recall"]
            f1 = report["1"]["f1-score"]
            pos_rate = np.mean(y_pred) * 100

            print(f"{t:<10.1f} {pos_rate:<8.1f} {prec:<10.2f} {rec:<10.2f} {f1:<10.2f}")


X_train, X_test, y_train, y_test = generate_data()

models = train_models(X_train, y_train)

results = evaluate_models(models, X_test, y_test)

threshold_analysis(models, X_test, y_test)

print("\nPODSUMOWANIE WYNIKÓW")
df = pd.DataFrame(results)
print(df.to_string(index=False))

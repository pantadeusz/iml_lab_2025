from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_and_display_results(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)


# Generuj dane niezbalansowane
X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42
)

# Bazowy model
model = LogisticRegression(random_state=42)

print("Bazowy model:")
train_and_display_results(model, X, y)

print()
# Z ważeniem klas
print("Model z ważeniem klas")
model_weighted = LogisticRegression(class_weight="balanced", random_state=42)
train_and_display_results(model_weighted, X, y)

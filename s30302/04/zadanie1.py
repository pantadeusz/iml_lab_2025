from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from dnn_model import accuracy

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc= accuracy_score(y_test, y_pred)

print(f"Porównanie wyników:")
print(f"Random Forest accuracy: {acc:.4f}")
print(f"DNN accuracy: {accuracy:.4f}")

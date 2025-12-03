from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import joblib

wine = fetch_ucirepo(id=109)

X = wine.data.features
y = wine.data.targets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(random_state=42)
model = classifier.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

joblib.dump(classifier, 'random_forest_classifier.pkl')
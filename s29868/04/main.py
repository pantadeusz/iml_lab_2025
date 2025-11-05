from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import joblib

mushroom = fetch_ucirepo(id=73)

X = mushroom.data.features
y = mushroom.data.targets
X = X.apply(LabelEncoder().fit_transform)
y = LabelEncoder().fit_transform(y.values.ravel())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(classification_report(y_test,y_pred))


matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(matrix).plot()
plt.show()

joblib.dump(rf, 'model.pkl')






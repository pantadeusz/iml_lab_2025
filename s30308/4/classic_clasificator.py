import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
print(path)
data = pd.read_csv(path + "/diabetes.csv")

# Set variables for the targets and features
y = data['Outcome']
X = data.drop('Outcome', axis=1)


# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

# Create the classifier and fit it to our training data
model = RandomForestClassifier(random_state=7, n_estimators=100)
model.fit(X_train, y_train)

# Predict classes given the validation features
y_pred = model.predict(X_test)

# Calculate the accuracy as our performance metric
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# Calculate the confusion matrix itself
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix--random_forest.png")
plt.close()
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import data

X_train, X_val, X_test, y_train, y_val, y_test = data.get_wine(simple=True)

def train_random_forest_classifier():
    
    # Create and train the Random Forest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Fit the model
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions on validation set
    y_pred = rf_classifier.predict(X_val)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Random Forest Classifier Validation Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Return the trained model and predictions
    return rf_classifier, y_pred

train_random_forest_classifier()

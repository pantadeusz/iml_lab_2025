from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



def train_and_display_results(model, X, y, resample=None):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    if resample == "smote":
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    elif resample == "under":
        under = RandomUnderSampler(random_state=42)
        X_train, y_train = under.fit_resample(X_train, y_train)
    else:
        print("Brak resamplingu.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))



X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42
)

# Bazowy model
print("\n--- Bazowy model (bez balansu) ---")
base_model = LogisticRegression(random_state=42)
train_and_display_results(base_model, X, y, resample=None)

# Model z ważeniem klas
print("\n--- Model z class_weight='balanced' ---")
weighted_model = LogisticRegression(class_weight="balanced", random_state=42)
train_and_display_results(weighted_model, X, y, resample=None)

# Oversampling (SMOTE)
print("\n--- Oversampling: SMOTE ---")
smote_model = LogisticRegression(random_state=42)
train_and_display_results(smote_model, X, y, resample="smote")

# Undersampling (RandomUnderSampler)
print("\n--- Undersampling: RandomUnderSampler ---")
under_model = LogisticRegression(random_state=42)
train_and_display_results(under_model, X, y, resample="under")

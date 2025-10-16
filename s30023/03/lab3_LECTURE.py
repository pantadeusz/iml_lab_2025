from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd
import kagglehub

def get_dataset():
    path = kagglehub.dataset_download("anishdevedward/loan-approval-dataset")
    df = pd.read_csv(path + "/loan_approval.csv")
    df.drop(['name', 'city'], axis=1, inplace=True)
    return df

def get_y_X(df):
    # X = df.iloc[:, :-1]
    # y = df.iloc[:, -1]
    X = df[['income', 'credit_score', 'loan_amount', 'years_employed', 'points']]
    y = df['loan_approved']
    return X, y

def test_classifier(models, X_train, X_test, y_train, y_test):
    plt.figure(figsize=(12, 5))

    for name, model in models:
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        # PR
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')

    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.show()

def use_model(input = [], model):
    # TODO
    return model.predict(input)

def main():
    X, y = get_y_X(get_dataset())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ('Logistic Regression', LogisticRegression(random_state=42)),
        ('Random Forest', RandomForestClassifier(random_state=42, max_depth=1, n_estimators=1)),
        ('SVM', SVC(probability=True, random_state=42))
    ]

    test_classifier(models, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
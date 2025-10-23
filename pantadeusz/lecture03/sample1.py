#  UWAGA: Poniższy kod jes specjalnie jednym wielkim bajzlem.
#         Nie nadaje sie on do użycia wprost.


import kagglehub
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve

path = kagglehub.dataset_download("anishdevedward/loan-approval-dataset")

df = pd.read_csv(f"{path}/loan_approval.csv")
df = df.drop(df.columns[0], axis=1)
original_columns = [ column_name for column_name in df.columns ]
df.loc[len(df)] = [None,38799,635,48259,17,40.0,False]
df.loc[len(df)] = [None,41957,763,16752,5,60.0,True]
df.iloc[:, 0] = df.iloc[:, 0].fillna('MISSING_PLACEHOLDER')

place_dummies = pd.get_dummies(df.iloc[:, 0], prefix='place')
df = df.drop(df.columns[0], axis=1)
df = pd.concat([df.iloc[:, :0], place_dummies, df.iloc[:, 0:]], axis=1)

X = df.iloc[:, :-1]
y = df.iloc[:, -1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('SVM', SVC(probability=True, random_state=42))
]

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

plt.savefig('plots.png')

# ################################################################# #
to_predict = [
['New Frank',38799,635,48259,17,40.0],
['East Haley',41957,763,16752,5,60.0],
]
df_to_predict = pd.DataFrame(to_predict, columns=original_columns[0:-1])


df_to_predict.iloc[:, 0] = df_to_predict.iloc[:, 0].fillna('MISSING_PLACEHOLDER')
print("DUMMIES",pd.get_dummies(df_to_predict.iloc[:, 0], prefix='place'))
dummies_predict = pd.get_dummies(df_to_predict.iloc[:, 0], prefix='place').reindex(columns=place_dummies.columns, fill_value=0)
print(len([i for i in dummies_predict.columns]))

df_to_predict = df_to_predict.drop(df_to_predict.columns[0], axis=1)
df_to_predict = pd.concat([dummies_predict, df_to_predict.iloc[:, 0:]], axis=1)

X_predict = df_to_predict.iloc[:, :]

print(X_predict)

for name, model in models:
    print(name)
    y_proba = model.predict_proba(X_predict)
    print(y_proba)
    y_pred = model.predict(X_predict)
    print(y_pred)


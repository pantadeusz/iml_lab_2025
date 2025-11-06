from keras.src.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, accuracy_score

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, TargetEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def prepare_data(data):
    df = pd.concat([data.data.features, data.data.targets], axis=1)
    df['income'] = df['income'].str.strip().str.replace('.', '', regex=False)

    # print("Columns:", df.columns)

    df = df.replace('?', np.nan)
    categorical_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    label_encoder = LabelEncoder()
    df['income'] = label_encoder.fit_transform(df['income'])

    edu_order = [['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th',
                  '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-acdm',
                  'Assoc-voc', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate']]
    ordinal_encoder = OrdinalEncoder(categories=edu_order)
    df['education'] = ordinal_encoder.fit_transform(df[['education']])

    nominal_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race']
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
    ohe_df = pd.DataFrame(
        one_hot_encoder.fit_transform(df[nominal_cols]),
        columns=one_hot_encoder.get_feature_names_out(nominal_cols),
        index=df.index
    )

    df['sex'] = np.where(df['sex'] == 'Male', 1, 0)

    target_encoder = TargetEncoder()
    df['native-country'] = target_encoder.fit_transform(df[['native-country']], df['income'])

    df_final = pd.concat([df.drop(nominal_cols, axis=1), ohe_df], axis=1)

    X = df_final.drop('income', axis=1)
    y = df_final['income']

    # print("Unique y values:", y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test

def predict_with_Random_Forest():
    model_random_forest = RandomForestClassifier(n_estimators=100, random_state=42, )
    model_random_forest.fit(X_train_encoded, y_train_encoded)
    return model_random_forest.predict(X_test_encoded)

def show_metrics(y_pred, y_true):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true,y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print("Random forest stats:")
    print(f"F1 score: {f1:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"accuracy: {accuracy:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig("confusion_matrix.png")
    plt.close()

def scale_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def plot_accuracy_and_validation(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy.png")
    plt.show()


    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss.png")
    plt.show()


data = fetch_ucirepo(id=2)

X_train_encoded,X_test_encoded, y_train_encoded, y_test_encoded = prepare_data(data)

random_forest_predictions = predict_with_Random_Forest()
show_metrics(random_forest_predictions, y_test_encoded)

X_train_scaled = scale_data(X_train_encoded)
X_test_scaled = scale_data(X_test_encoded)

model = models.Sequential([
    tf.keras.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train_encoded,
    validation_data=(X_test_scaled, y_test_encoded),
    epochs=50,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

plot_accuracy_and_validation(history)


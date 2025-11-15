import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import keras_tuner as kt
import os


def my_print(s):
    print("-" * 50, s, "-" * 50)

def print_model_report(y_true, y_pred):
    cls_report = classification_report(y_true, y_pred, output_dict=True)
    cf_matrix = confusion_matrix(y_true, y_pred)
    my_print("Classification Report")
    print(cls_report)
    my_print("Confusion Matrix")
    print(cf_matrix)

def get_train_test_data():
    # https://www.kaggle.com/datasets/prince2004patel/iti-student-dropout-synthetic-dataset
    data = pd.read_csv("iti_student_dropout_dataset.csv")
    X = data.drop("dropout", axis=1)
    y = data["dropout"].map({"Yes": 1, "No": 0})

    numerical_cols = [
        "age",
        "family_income",
        "distance_to_institute",
        "tenth_marks",
        "attendance_rate",
        "test_scores_avg",
        "practical_skills_rating",
        "backlogs",
        "teaching_quality_rating",
        "motivation_score",
        "num_siblings",
    ]

    categorical_cols = [
        "gender",
        "location_type",
        "program_enrolled",
        "financial_aid_status",
        "part_time_work",
        "career_alignment",
        "family_support",
        "stress_levels",
        "internet_connectivity_issues",
        "parents_education",
        "ragging_experience",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test


def run_rfc_model(X_train, y_train, X_test, y_test):
    my_print("Model Random Forest Classifier")
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    print_model_report(y_test, y_pred)


def build_model(hp, input_shape):
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=hp.Int("units_0", min_value=32, max_value=512, step=32),
            activation=hp.Choice("activation", ["relu", "tanh"]),
            input_shape=input_shape,
        )
    )

    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"units_{i+1}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )

    if hp.Boolean("dropout"):
        model.add(keras.layers.Dropout(rate=0.25))

    model.add(keras.layers.Dense(1, activation="sigmoid"))
    lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def save_model(model, save_model_base):
    extension = ".keras"
    counter = 1
    save_model_path = f"{save_model_base}_{counter}{extension}"

    while os.path.exists(save_model_path):
        save_model_path = f"{save_model_base}_{counter}{extension}"
        counter += 1

    model.save(save_model_path)

def main():
    X_train, X_test, y_train, y_test = get_train_test_data()

    run_rfc_model(X_train, y_train, X_test, y_test)

    epochs = 30
    max_trials = 2
    executions_per_trial = 5

    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, input_shape=(X_train.shape[1],)),
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory="iti_dropout_tuning_",
        overwrite=True,
        project_name=f"{epochs}_epochs_{max_trials}_trials_{executions_per_trial}_executions_per_trial",
    )

    # tuner.search_space_summary()

    tuner.search(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=0,
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()

    y_pred_proba = best_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype("int32")
    my_print("Best Neural Network Model")
    print_model_report(y_test, y_pred)

    save_model_base = f"./models/{epochs}_{max_trials}_{executions_per_trial}"
    save_model(best_model, save_model_base)

if __name__ == "__main__":
    main()

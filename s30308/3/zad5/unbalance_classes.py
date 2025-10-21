"""
Notatki z zadania
------------------------------------------------------------------------------------------------------------------------

Niezbalansowane klasy powodują, że model może bardzo dobrze przewidywać przypadki z klasy dominującej,
a jednocześnie ignorować lub źle klasyfikować klasę mniejszościową.
Nawet jeśli dana klasa występuje rzadko, w wielu zastosowaniach jest krytyczna, dlatego model musi umieć ją wykrywać.

------------------------------------------------------------------------------------------------------------------------

Overfitting - Overfitting to sytuacja, w której model uczy się zbyt dokładnie danych treningowych, w tym szumu,
wyjątków i przypadkowych fluktuacji, zamiast uogólniać prawdziwe wzorce.
W skrócie: model dostosowuję się pod dane treningowe

------------------------------------------------------------------------------------------------------------------------

Są trzy główne metody z problemem niezbalansowanych klas:

------------------------------------------------------------------------------------------------------------------------

1. Ważenie klas (class_weight)
Metoda polega na dodaniu większej wagi dla klasy rzadszej. Dzięki temu model "przekłada większą uwagę na tą klasę".
W sklearn przed inicjalizacją modelu możemy ustawić rodzaj przypisanie wag.
class_weight='balanced' → wagi liczone jako n_samples / (n_classes * n_samples_class)

Zalety:
- Prosta implementacja
- Nie zmienia danych, działa bez modyfikacji zbioru

Wady:
- Nie zawsze wystarcza przy bardzo mocnym niezbalansowaniu
- Może prowadzić do lekkiego overfittingu dla klasy mniejszościowej

---------------------------------------------------------------------------------------------------

2. Oversampling - SMOTE (Synthetic Minority Oversampling Technique)
Tworzy sztuczne próbki dla rzadszej klasy zamiast powtarzać istniejące.

Jak to działa?
1. Dla każdej próbki klasy mniejszościowej wybieramy k najbliższych sąsiadów z tej samej klasy.
2. Losowo wybieramy sąsiada i generujemy nową próbkę na linii między nimi w przestrzeni cech.

Wymaga imblearn

Zalety:
- Poprawia recall i F1-score dla klasy rzadkiej
- Tworzy różnorodne próbki, mniej overfittingu niż zwykłe kopiowanie

Wady:
- Wolniejsze przy dużych zbiorach
- Ryzyko wprowadzenia „nienaturalnych” punktów, jeśli cech jest bardzo dużo

---------------------------------------------------------------------------------------------------

3. Undersampling (losowe zmienianie klasy dominującej)
Zmniejszamy liczbę przykładów klasy dominującej, żeby bilans między klasami był lepszy.
Dzięki temu model nie jest zdominowany przez większość, łatwiej nauczy się klasy rzadkiej.

Jak to działa?
1. Losowo wybieramy tylko część przykładów klasy dominującej.
2. Łączymy je z pełną klasą mniejszościową.
3. Model trenujemy na nowym, zbalansowanym zbiorze.

Zalety:
- Szybsze trenowanie modelu (mniej danych)
- Redukuje dominację klasy większościowej

Wady:
- Utrata informacji z klasy dominującej
- Może pogorszyć ogólną dokładność modelu

------------------------------------------------------------------------------------------------------------------------
"""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def generate_confusion_matrix(prediction, model_name="unknown"):
    # Macierz pomyłek
    cm = confusion_matrix(y_test, prediction)

    # Raport klasyfikacji
    report_dict = classification_report(y_test, prediction, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Rysowanie w matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Macierz pomyłek
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=axes[0], colorbar=False)
    axes[0].set_title(f"Confusion Matrix\n{model_name}")

    # 2. Raport klasyfikacji jako tabela
    axes[1].axis('off')  # wyłącz oś
    axes[1].table(cellText=report_df.round(2).values,
                  colLabels=report_df.columns,
                  rowLabels=report_df.index,
                  loc='center')
    axes[1].set_title(f"Classification Report\n{model_name}")

    plt.tight_layout()
    plt.savefig(f"confusion_matrix--{model_name}.png")
    plt.close()

# Generuj dane niezbalansowane
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42)

# Bazowy model
model = LogisticRegression(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
generate_confusion_matrix(y_pred, "base_model")


# 1. Ważenie klas
model_weighted = LogisticRegression(class_weight='balanced', random_state=42)
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)
generate_confusion_matrix(y_pred_weighted, "weighted_model")

# 2. SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

model_smote = LogisticRegression(random_state=42)
model_smote.fit(X_smote, y_smote)
y_pred_smote = model_smote.predict(X_test)
generate_confusion_matrix(y_pred_smote, "smote_model")

# 3. Undersampling
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)

model_under = LogisticRegression(random_state=42)
model_under.fit(X_rus, y_rus)
y_pred_rus = model_under.predict(X_test)
generate_confusion_matrix(y_pred_rus, "undersampling_model")
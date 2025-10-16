import os
import pandas as pd
import shutil
import kagglehub

# Pobierz dataset (trafi do cache, ale zwróci ścieżkę do pliku CSV)
path = kagglehub.dataset_download("mirzayasirabdullah07/student-exam-scores-dataset")

target_dir = os.getcwd()

# Skopiuj CSV do obecnego katalogu
src = os.path.join(path, "student_exam_scores.csv")
dst = os.path.join(target_dir, "student_exam_scores.csv")
shutil.copy(src, dst)

df = pd.read_csv("student_exam_scores.csv")
print(df.head())
import kagglehub
import os
import pandas as pd
import tensorflow as tf

def download_dataset():
    path = kagglehub.dataset_download("ahmeduzaki/earthquake-alert-prediction-dataset")
    path = os.path.join(path, 'earthquake_alert_balanced_dataset.csv')
    return path

def test(path):
    data = pd.read_csv(path)
    print(data.head())

if __name__ == '__main__':
    path = download_dataset()
    test(path)
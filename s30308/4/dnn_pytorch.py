"""
------------------------------------------------------------------------------------------------------------------------

Tutorial:
https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/

Budowa sieci neuronowej:

Pierwszą rzeczą, którą należy zrobić, aby uzyskać prawidłowy wynik, jest upewnienie się,
że pierwsza warstwa ma prawidłową liczbę cech wejściowych. W tym przykładzie można określić wymiar wejściowy (Input) 8
dla ośmiu zmiennych wejściowych jako jeden wektor.

Dalsze paremetry w innych warstawach może ustalić za pomocą heurystyk lub za pomocą prób i błędów. Ogólnym celem
jest stworzeni sieci, która jest wystarczająca duża, aby objąć pewien problem oraz wystarczająco mała, aby była szybka.

Warstwy w pełni połączone lub warstwy gęste definiuje się za pomocą klasy Linear w PyTorch. Oznacza to po prostu
operację podobną do mnożenia macierzy. Jako pierwszy argument można podać liczbę wejść,
a jako drugi argument liczbę wyjść. Liczba wyjść jest czasami nazywana liczbą neuronów lub liczbą węzłów w warstwie.

Potrzebna jest również funkcja aktywacji po warstwie. Jeśli nie zostanie podana, wystarczy przenieść wynik mnożenia
macierzy do następnego kroku lub czasami wywołać go za pomocą aktywacji liniowej, stąd nazwa warstwy (layer).

Aby stworzyć obiekt modelu musimy utworzyć klasę, która dziedziczy po nn.Module.
Klasa musi mieć implementację metody forward

============================================================================

Przygotowanie do treningu

Kiedy już zdefiniowaliśmy model wystarczy go wytrenować, ale musimy ocenić cel szkolenia.
Chcemy, aby model sieci neuronowej generował wynik jak najbardziej zbliżony do y. Szkolenie sieci oznacza znalezienie
najlepszego zestawu wag do mapowania danych wejściowych na dane wyjściowe w zbiorze danych.
Funkcja straty jest miarą odległości prognozy od y. W tym przykładzie należy użyć binarnej entropii krzyżowej,
ponieważ jest to problem klasyfikacji binarnej (0/1).

todo dokończyć pytorch + tenserflow

------------------------------------------------------------------------------------------------------------------------

Funkcja ReLU (ang. Rectified Linear Unit) to jedna z najczęściej używanych funkcji aktywacji w sieciach neuronowych,
zwłaszcza w sieciach głębokiego uczenia (deep learning).

ReLU przepuszcza tylko dodatnie wartości, a wszystkie ujemne „zeruje”.
Dzięki temu:
- Model wprowadza nieliniowość (niezbędną do nauki złożonych wzorców),
- Obliczenia pozostają proste i szybkie.


------------------------------------------------------------------------------------------------------------------------

"""

import kagglehub
import os
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Download latest version
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
print(path)
data = pd.read_csv(path + "/diabetes.csv")
print(data.columns)


# Dzielimy na atrybuty i etykiety
y = data['Outcome']
X = data.drop('Outcome', axis=1)

# Konwertujemy dane na tensory
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)


class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

model = PimaClassifier()
print(model)


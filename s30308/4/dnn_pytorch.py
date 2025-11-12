"""
------------------------------------------------------------------------------------------------------------------------

Tutorial:
https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/

============================================================================

Budowa sieci neuronowej

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

Kiedy zdecydujemy funkcję straty potrzebny jest optymalizator. Jest to algorytm, który aktualizuje wagi sieci na
podstawie gradientów (czyli kierunku, w którym trzeba „zejść” po powierzchni błędu, żeby znaleźć minimum straty).

------------------------------------------------------------------------------------------------------------------------

Funkcja ReLU (ang. Rectified Linear Unit) to jedna z najczęściej używanych funkcji aktywacji w sieciach neuronowych,
zwłaszcza w sieciach głębokiego uczenia (deep learning).

ReLU przepuszcza tylko dodatnie wartości, a wszystkie ujemne „zeruje”.
Dzięki temu:
- Model wprowadza nieliniowość (niezbędną do nauki złożonych wzorców),
- Obliczenia pozostają proste i szybkie.

------------------------------------------------------------------------------------------------------------------------

Binarna entropia krzyżowa (ang. binary cross-entropy, BCE) to funkcja straty wykorzystywana w uczeniu modeli
klasyfikacji binarnej – czyli wtedy, gdy wynik może być 0 albo 1. W praktyce mierzy,
jak bardzo przewidywania modelu różnią się od rzeczywistych etykiet. Funkcja działa tak, że karze mocniej wyniki
dalsze od zamierzonego.
Przykład:
Jeśli rzeczywista etykieta to 1, a model prawdopodobieństwo przewiduje 0.9, to model jest prawie pewny i ma rację,
więc kara (czyli strata) będzie bardzo mała.
Ale jeśli przewidzi 0.49, to oznacza, że jest prawie niepewny — czyli połowicznie myśli, że to 1,
a połowicznie że to 0 — więc kara będzie znacznie większa.

------------------------------------------------------------------------------------------------------------------------

Optymalizator Adam - (skrót od Adaptive Moment Estimation)to jeden z najpopularniejszych i najskuteczniejszych
optymalizatorów w uczeniu sieci neuronowych. Działa bardzo dobrze, ale ma problemy — zwłaszcza przy dużych
zbiorach danych i skomplikowanych krajobrazach błędu.

Jak mniej więcej działa?
Adam dynamicznie dostosowuje krok uczenia się dla każdej wagi osobno, wykorzystując średnią z gradientów i ich kwadratów.

============================================================================

Trenowanie modelu

Po przygotowaniu model możemy zabrać się za trenowanie. Proces trenowania odbywa się epokach i batchach

Epoch - jedno przejście całego zbioru danych treningowych przez model

Batch - zbiór próbek treningowych, które model przetwarza jednocześnie w jednej iteracji aktualizacji wag.
Uczenie na całym zbiorze (tzw. Batch Gradient Descent) byłoby bardzo dokładne, ale wolne i pamięciożerne —
szczególnie przy milionach przykładów. Dlatego stosuje się mniejsze partie

Rozmiar partii jest ograniczony przez pamięć systemu. Ponadto liczba wymaganych obliczeń jest
liniowo proporcjonalna do rozmiaru partii. Całkowita liczba partii w wielu epokach to liczba uruchomień
algorytmu gradientu prostego w celu udoskonalenia modelu.
Jest to kompromis – potrzeba więcej iteracji algorytmu gradientu prostego, aby uzyskać lepszy model,
ale jednocześnie nie chcemy, aby ukończenie treningu trwało zbyt długo.
Liczbę epok i rozmiar partii można dobrać eksperymentalnie metodą prób i błędów.


============================================================================

Ewaluacja modelu

To etap sprawdzania, jak dobrze działa wytrenowany model na danych,
których nie widział podczas treningu (czyli na danych testowych lub walidacyjnych).
Celem jest sprawdzenie, czy model nauczył się uogólniać, a nie tylko zapamiętywać dane treningowe.


============================================================================



"""

import kagglehub
import torch
from torch import nn
import pandas as pd
import torch.optim as optim

# Download latest version
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
data = pd.read_csv(path + "/diabetes.csv")


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

loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001) # w pierwszy argumencie podajemy co optymalizujemy (czyli model)

n_epochs = 300
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size): #range(start, stop, step
        X_batch = X[i: i + batch_size] # slicing - od końca starego batch'a do końca nowego
        y_pred = model(X_batch) # funkcja model automatycznie wywołuje forward()
        y_batch = y[i:i + batch_size] # pobieramy etykiety do obliczenia funkcji straty
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad() # zerujemy gradient, aby się nie sumował wraz z kolejnymi iteracjami
        loss.backward() # obliczanie gradientu (pochodne)
        optimizer.step() # Optymalizator (tu: Adam) wykonuje aktualizację wag na podstawie gradientów obliczonych w poprzednim kroku.
                         # Bez tego model się nie będzie uczyć

    print(f'Finished epoch {epoch}, latest loss {loss}')

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)

accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")
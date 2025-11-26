# Podsumowanie eksperymentów - Wine Classification

## Dotychczasowe wyniki

| Model | Architektura | Epochs | Batch Size | Regularyzacja | Rozmiar (bytes) | Accuracy    |
|-------|--------------|--------|------------|--------------|-----------------|-------------|
| Random Forest | 100 drzew    | -      | -          | -            | 211,569         | **100.00%** |
| NN Basic | 32-16-3      | 50     | 16         | Brak         | 37,745          | 33.33%      |
| NN Normalized | 64-32-3      | 50     | 16         | Brak         | 65,803          | 94.44%      |
| NN Normalized | 64-32-3      | 80     | 16         | Brak         | 65,811          | 97.22%      |
| NN Normalized | 128-64-3     | 50     | 16         | Brak         | 151,829         | 94.44%      |
| NN Normalized | 128-64-3     | 80     | 16         | Brak         | 151,831         | 94.44%      |
| NN Normalized | 64-32-3      | 50     | 16         | Dropout      | 70,115          | 97.22%      |
| NN Normalized | 64-32-3      | 80     | 16         | Dropout      | 70,125          | 97.22%      |
| NN Normalized | 64-32-3      | 50     | 8          | Brak         | 65,803          | 94.44%      |
| NN Normalized | 64-32-3      | 80     | 8          | Brak         | 65,811          | 97.22%      |
| NN Normalized | 32-16-3      | 100    | 8          | Brak         | 41,227          | 97.22%      |
| NN Normalized | 32-16-3      | 200    | 8          | Brak         | 41,235          | 94.44%      |
| NN Normalized | 16-8-3       | 50     | 8          | Dropout(0.2) | 44,159          | **100.00%** |


Najlepszym rozwiązaniem okazała się znormalizowana sieć neuronowa 
o małej architekturze (16 i 8 neuronów) trenowana przez 50 epok z regularyzacją Dropout (0.2).

Dataset Wine jest małym zbiorem danych (tylko 178 próbek). Eksperymenty pokazały, że 
duże sieci (np. 128-64) nie przynosiły poprawy, a często wręcz radziły sobie gorzej (94.44%).

Zastosowanie warstwy Dropout(0.2) w zwycięskim modelu pozwoliło na osiągnięcie 100% na zbiorze walidacyjnym. 
Wersje bez dropoutu (nawet przy większej liczbie epok) często zatrzymywały się na 97.22%, co sugeruje, że dropout pomógł w 
lepszej generalizacji.

Zwycięski model używał batch_size=8 (w przeciwieństwie do 16 w innych próbach). Przy tak małym zbiorze danych, mniejszy rozmiar 
partii (batch) często pozwala na częstsze aktualizacje wag i dokładniejsze "przeszukiwanie" powierzchni funkcji błędu, co pomogło 
wycisnąć ostatnie procenty skuteczności.


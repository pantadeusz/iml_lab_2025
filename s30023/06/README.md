# Wyniki
Testowałem różne metode aktywacji (relu, elu, tahn), optymilizacji (adam, sgd, rmsrop), inicjalizacji (glorot_uniform, he_normal, random_normal), ilośc neuronów na warstwie 1 i 2, learning rate

- Najlepszy optymilizator - Adam
- Najlepszy learning rate - 0.001
- Najlepsza funkcja aktywacji - relu (dość często też wychodził elu)
- Najlepsza funkcja inicjalizacji - glorot_uniform (czasami he_normal)
- Najlepsza ilość neuronów w pierwszej warstwie - pomiędzy od 32 do 96
- Najlepsza ilość neuronów w drugiej warstwie - pomiędzy od 16 do 64

# Problemy które spotkałem przy rozwiązywaniu
### Error [Process finished with exit code 139 (interrupted by signal 11:SIGSEGV)]

Problem wynikał przez to że tensorflow chciał jakiejś informacji z dyrektorij do których nie miał dostępu

Rozwiązanie:
```
conda install -c apple tensorflow-deps -y

pip install tensorflow-macos
pip install tensorflow-metal
```


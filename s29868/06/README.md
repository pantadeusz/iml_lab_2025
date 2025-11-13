# Podsumowanie wyników modeli dla klasyfikacji roślin fasoli

| Model          | Test accuracy | Przykładowe predykcje              | Najlepsze hiperparametry / uwagi                          |
|----------------|---------------|-----------------------------------|-----------------------------------------------------------|
| default_model  | 45.3%         | healthy.jpg → angular_leaf_spot (90.85%) | Prosty MLP bez strojenia                                  |
| tuned_model    | 45.3%         | healthy.jpg → healthy (34.06%)    | units1: 128–256, units2: 32–128, activation1: tanh, activation2: relu/sigmoid, kernel_initializer: glorot_uniform / he_normal, optimizer: adam / sgd / rmsprop, lr: 1e-4–1e-2 |
| cnn_model      | 69.5%         | healthy.jpg → healthy (98.22%)<br>rust.jpg → bean_rust (77.50%) | CNN: 3x Conv2D + MaxPooling, Dropout, Dense(128)+Dropout, Dense(3, softmax), optimizer: Adam, learning_rate domyślne |

# Wnioski
- MLP daje ograniczoną dokładność (~45–63%), nawet po strojeniu hiperparametrów.
- CNN znacząco poprawia dokładność (~70%), dobrze przewiduje klasy z dużą pewnością.
- Najlepszy model do dalszych eksperymentów: CNN.
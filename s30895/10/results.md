
baseline:\
Test Loss: 0.32919543981552124\
Test Accuracy: 0.8637199997901917

    Value             |Hyperparameter
    64                |embedding_dim
    64                |lstm_1_units
    32                |lstm_2_units
    64                |dense_units
    0.5               |dropout
    0.0001            |learning_rate

    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    │ text_vectorization              │ (None, None)           │             0 │
    │ (TextVectorization)             │                        │               │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ embedding (Embedding)           │ (None, None, 64)      │       128,000 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ bidirectional (Bidirectional)   │ (None, None, 64)      │        98,816 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ bidirectional_1 (Bidirectional) │ (None, 32)            │        98,816 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense (Dense)                   │ (None, 64)             │         8,256 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout (Dropout)               │ (None, 64)             │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense_1 (Dense)                 │ (None, 1)              │            65 │
    └─────────────────────────────────┴────────────────────────┴───────────────┘

tuned:\
Test Loss: 0.32221415638923645
Test Accuracy: 0.8660399913787842

    Value             |Hyperparameter
    128                |embedding_dim
    64                |lstm_1_units
    64                |lstm_2_units
    64                |dense_units
    0.4               |dropout
    0.000143          |learning_rate


    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    │ text_vectorization              │ (None, None)           │             0 │
    │ (TextVectorization)             │                        │               │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ embedding (Embedding)           │ (None, None, 128)      │       128,000 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ bidirectional (Bidirectional)   │ (None, None, 128)      │        98,816 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ bidirectional_1 (Bidirectional) │ (None, 128)            │        98,816 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense (Dense)                   │ (None, 64)             │         8,256 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout (Dropout)               │ (None, 64)             │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense_1 (Dense)                 │ (None, 1)              │            65 │
    └─────────────────────────────────┴────────────────────────┴───────────────┘

wnioski:
tuner znalazł nieznacznie lepszą architekturę w stosunku do bazowej. W wyniku dostosowywania atchitekrtury\
zwiększony został nieznacznie learning rate oraz w warstwach embedding i lstm zwiększona została wielkość wektorów.\
Dropout został nieznacznie zmniejszony do 0.4.
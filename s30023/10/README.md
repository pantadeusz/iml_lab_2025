# Problemy z którymi spotkałem się podczas wykonywania zadania

- Warstwa LSTM nie działa poprawnie na Mac-ach i zawsze zdarza się problem z leakage of data, podczas którego kernel_task zaczyna zajmować pod 80 GB (kiedy RAM-u mam tylko 24 GB)
- Jedynym rozwiązaniem tego problemu było zrobienie zadania na maszynie wirtualnej

# Logi trenowania z Hyperband 

```
Search: Running Trial #1

Value             |Best Value So Far |Hyperparameter
128               |128               |lstm_units_1
16                |16                |lstm_units_2
112               |112               |dense_units
0.001             |0.001             |learning_rate
0.4               |0.4               |drop_amount
2                 |2                 |tuner/epochs
0                 |0                 |tuner/initial_epoch
1                 |1                 |tuner/bracket
0                 |0                 |tuner/round

Epoch 1/2
2025-12-14 11:17:50.609318: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:473] Loaded cuDNN version 91500
391/391 ━━━━━━━━━━━━━━━━━━━━ 67s 153ms/step - accuracy: 0.5322 - loss: 0.6743 - val_accuracy: 0.5212 - val_loss: 0.6909
Epoch 2/2
391/391 ━━━━━━━━━━━━━━━━━━━━ 60s 153ms/step - accuracy: 0.6562 - loss: 0.6047 - val_accuracy: 0.7420 - val_loss: 0.5254

Trial 1 Complete [00h 02m 07s]
val_accuracy: 0.7419599890708923

Best val_accuracy So Far: 0.7419599890708923
Total elapsed time: 00h 02m 07s

Search: Running Trial #2

Value             |Best Value So Far |Hyperparameter
64                |128               |lstm_units_1
48                |16                |lstm_units_2
48                |112               |dense_units
0.01              |0.001             |learning_rate
0.2               |0.4               |drop_amount
2                 |2                 |tuner/epochs
0                 |0                 |tuner/initial_epoch
1                 |1                 |tuner/bracket
0                 |0                 |tuner/round

Epoch 1/2
391/391 ━━━━━━━━━━━━━━━━━━━━ 65s 156ms/step - accuracy: 0.7344 - loss: 0.5013 - val_accuracy: 0.8437 - val_loss: 0.3715
Epoch 2/2
391/391 ━━━━━━━━━━━━━━━━━━━━ 61s 155ms/step - accuracy: 0.8414 - loss: 0.3536 - val_accuracy: 0.8520 - val_loss: 0.3163

Trial 2 Complete [00h 02m 27s]
val_accuracy: 0.8519999980926514

Best val_accuracy So Far: 0.8519999980926514
Total elapsed time: 00h 04m 34s

Search: Running Trial #3

Value             |Best Value So Far |Hyperparameter
96                |64                |lstm_units_1
64                |48                |lstm_units_2
48                |48                |dense_units
0.0001            |0.01              |learning_rate
0.4               |0.2               |drop_amount
2                 |2                 |tuner/epochs
0                 |0                 |tuner/initial_epoch
1                 |1                 |tuner/bracket
0                 |0                 |tuner/round

Epoch 1/2
391/391 ━━━━━━━━━━━━━━━━━━━━ 66s 158ms/step - accuracy: 0.6284 - loss: 0.5891 - val_accuracy: 0.8054 - val_loss: 0.4015
Epoch 2/2
391/391 ━━━━━━━━━━━━━━━━━━━━ 81s 155ms/step - accuracy: 0.8363 - loss: 0.3701 - val_accuracy: 0.8512 - val_loss: 0.3312

Trial 3 Complete [00h 02m 48s]
val_accuracy: 0.8512399792671204

Best val_accuracy So Far: 0.8519999980926514
Total elapsed time: 00h 07m 23s

Search: Running Trial #4

Value             |Best Value So Far |Hyperparameter
128               |64                |lstm_units_1
48                |48                |lstm_units_2
128               |48                |dense_units
0.001             |0.01              |learning_rate
0.3               |0.2               |drop_amount
2                 |2                 |tuner/epochs
0                 |0                 |tuner/initial_epoch
1                 |1                 |tuner/bracket
0                 |0                 |tuner/round

Epoch 1/2
391/391 ━━━━━━━━━━━━━━━━━━━━ 67s 161ms/step - accuracy: 0.6881 - loss: 0.5529 - val_accuracy: 0.8306 - val_loss: 0.4156
Epoch 2/2
391/391 ━━━━━━━━━━━━━━━━━━━━ 62s 158ms/step - accuracy: 0.8186 - loss: 0.3926 - val_accuracy: 0.8418 - val_loss: 0.3880

Trial 4 Complete [00h 02m 09s]
val_accuracy: 0.8417999744415283

Best val_accuracy So Far: 0.8519999980926514
Total elapsed time: 00h 09m 32s

Search: Running Trial #5

Value             |Best Value So Far |Hyperparameter
96                |64                |lstm_units_1
64                |48                |lstm_units_2
64                |48                |dense_units
0.01              |0.01              |learning_rate
0.5               |0.2               |drop_amount
2                 |2                 |tuner/epochs
0                 |0                 |tuner/initial_epoch
1                 |1                 |tuner/bracket
0                 |0                 |tuner/round

Epoch 1/2
391/391 ━━━━━━━━━━━━━━━━━━━━ 69s 165ms/step - accuracy: 0.6615 - loss: 0.5701 - val_accuracy: 0.8417 - val_loss: 0.3797
Epoch 2/2
391/391 ━━━━━━━━━━━━━━━━━━━━ 78s 154ms/step - accuracy: 0.8320 - loss: 0.3732 - val_accuracy: 0.7918 - val_loss: 0.4010

Trial 5 Complete [00h 02m 27s]
val_accuracy: 0.8416799902915955

Best val_accuracy So Far: 0.8519999980926514
Total elapsed time: 00h 11m 59s

Search: Running Trial #6

Value             |Best Value So Far |Hyperparameter
64                |64                |lstm_units_1
48                |48                |lstm_units_2
48                |48                |dense_units
0.01              |0.01              |learning_rate
0.2               |0.2               |drop_amount
6                 |2                 |tuner/epochs
2                 |0                 |tuner/initial_epoch
1                 |1                 |tuner/bracket
1                 |0                 |tuner/round
0001              |None              |tuner/trial_id

/home/iml/iml_lab_2025/s30023/10/venv/lib/python3.12/site-packages/keras/src/saving/saving_lib.py:797: UserWarning: Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 36 variables.
  saveable.load_own_variables(weights_store.get(inner_path))
Epoch 3/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 65s 157ms/step - accuracy: 0.8598 - loss: 0.3151 - val_accuracy: 0.8681 - val_loss: 0.3043
Epoch 4/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 60s 153ms/step - accuracy: 0.8713 - loss: 0.2847 - val_accuracy: 0.8615 - val_loss: 0.2971
Epoch 5/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 60s 154ms/step - accuracy: 0.8850 - loss: 0.2599 - val_accuracy: 0.8667 - val_loss: 0.2908
Epoch 6/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 60s 154ms/step - accuracy: 0.8922 - loss: 0.2451 - val_accuracy: 0.8564 - val_loss: 0.2987

Trial 6 Complete [00h 04m 28s]
val_accuracy: 0.8680800199508667

Best val_accuracy So Far: 0.8680800199508667
Total elapsed time: 00h 16m 27s

Search: Running Trial #7

Value             |Best Value So Far |Hyperparameter
96                |64                |lstm_units_1
64                |48                |lstm_units_2
48                |48                |dense_units
0.0001            |0.01              |learning_rate
0.4               |0.2               |drop_amount
6                 |6                 |tuner/epochs
2                 |2                 |tuner/initial_epoch
1                 |1                 |tuner/bracket
1                 |1                 |tuner/round
0002              |0001              |tuner/trial_id

Epoch 3/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 65s 157ms/step - accuracy: 0.8514 - loss: 0.3353 - val_accuracy: 0.7748 - val_loss: 0.3822
Epoch 4/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 62s 157ms/step - accuracy: 0.8554 - loss: 0.3296 - val_accuracy: 0.8279 - val_loss: 0.3381
Epoch 5/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 61s 155ms/step - accuracy: 0.8631 - loss: 0.3176 - val_accuracy: 0.8541 - val_loss: 0.3190
Epoch 6/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 63s 160ms/step - accuracy: 0.8635 - loss: 0.3155 - val_accuracy: 0.8286 - val_loss: 0.3369

Trial 7 Complete [00h 04m 10s]
val_accuracy: 0.8541200160980225

Best val_accuracy So Far: 0.8680800199508667
Total elapsed time: 00h 20m 37s

Search: Running Trial #8

Value             |Best Value So Far |Hyperparameter
96                |64                |lstm_units_1
48                |48                |lstm_units_2
112               |48                |dense_units
0.0001            |0.01              |learning_rate
0.5               |0.2               |drop_amount
6                 |6                 |tuner/epochs
0                 |2                 |tuner/initial_epoch
0                 |1                 |tuner/bracket
0                 |1                 |tuner/round

Epoch 1/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 69s 166ms/step - accuracy: 0.6210 - loss: 0.5933 - val_accuracy: 0.8268 - val_loss: 0.3775
Epoch 2/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 61s 155ms/step - accuracy: 0.8379 - loss: 0.3644 - val_accuracy: 0.8194 - val_loss: 0.3539
Epoch 3/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 61s 155ms/step - accuracy: 0.8568 - loss: 0.3302 - val_accuracy: 0.8534 - val_loss: 0.3183
Epoch 4/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 81s 153ms/step - accuracy: 0.8598 - loss: 0.3228 - val_accuracy: 0.8441 - val_loss: 0.3248
Epoch 5/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 62s 160ms/step - accuracy: 0.8630 - loss: 0.3147 - val_accuracy: 0.8602 - val_loss: 0.3248

Trial 8 Complete [00h 05m 34s]
val_accuracy: 0.8601999878883362

Best val_accuracy So Far: 0.8680800199508667
Total elapsed time: 00h 26m 12s

Search: Running Trial #9

Value             |Best Value So Far |Hyperparameter
64                |64                |lstm_units_1
16                |48                |lstm_units_2
112               |48                |dense_units
0.001             |0.01              |learning_rate
0.5               |0.2               |drop_amount
6                 |6                 |tuner/epochs
0                 |2                 |tuner/initial_epoch
0                 |1                 |tuner/bracket
0                 |1                 |tuner/round

Epoch 1/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 63s 152ms/step - accuracy: 0.7103 - loss: 0.5324 - val_accuracy: 0.8306 - val_loss: 0.3930
Epoch 2/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 58s 148ms/step - accuracy: 0.8166 - loss: 0.4030 - val_accuracy: 0.8505 - val_loss: 0.3520
Epoch 3/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 58s 148ms/step - accuracy: 0.8400 - loss: 0.3537 - val_accuracy: 0.8563 - val_loss: 0.3231
Epoch 4/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 58s 149ms/step - accuracy: 0.8507 - loss: 0.3363 - val_accuracy: 0.8508 - val_loss: 0.3802
Epoch 5/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 58s 149ms/step - accuracy: 0.8636 - loss: 0.3186 - val_accuracy: 0.8466 - val_loss: 0.3238

Trial 9 Complete [00h 04m 57s]
val_accuracy: 0.8563200235366821

Best val_accuracy So Far: 0.8680800199508667
Total elapsed time: 00h 31m 09s

Search: Running Trial #10

Value             |Best Value So Far |Hyperparameter
32                |64                |lstm_units_1
64                |48                |lstm_units_2
32                |48                |dense_units
0.0001            |0.01              |learning_rate
0.3               |0.2               |drop_amount
6                 |6                 |tuner/epochs
0                 |2                 |tuner/initial_epoch
0                 |1                 |tuner/bracket
0                 |1                 |tuner/round

Epoch 1/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 64s 155ms/step - accuracy: 0.6298 - loss: 0.5899 - val_accuracy: 0.8250 - val_loss: 0.4161
Epoch 2/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 62s 158ms/step - accuracy: 0.8232 - loss: 0.3901 - val_accuracy: 0.8320 - val_loss: 0.3499
Epoch 3/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 59s 151ms/step - accuracy: 0.8476 - loss: 0.3485 - val_accuracy: 0.8577 - val_loss: 0.3431
Epoch 4/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 58s 149ms/step - accuracy: 0.8515 - loss: 0.3398 - val_accuracy: 0.8428 - val_loss: 0.3284
Epoch 5/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 61s 155ms/step - accuracy: 0.8570 - loss: 0.3302 - val_accuracy: 0.8556 - val_loss: 0.3223
Epoch 6/6
391/391 ━━━━━━━━━━━━━━━━━━━━ 61s 156ms/step - accuracy: 0.8598 - loss: 0.3179 - val_accuracy: 0.8544 - val_loss: 0.3189

Trial 10 Complete [00h 06m 06s]
val_accuracy: 0.8576800227165222

Best val_accuracy So Far: 0.8680800199508667
Total elapsed time: 00h 37m 15s

            Best HPs:
            LSTM 1 units: 64
            LSTM 2 units: 48
            Dense units: 48
            Learning rate: 0.01
            Drop amount: 0.2


Epoch 1/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 64s 154ms/step - accuracy: 0.6880 - loss: 0.5583 - val_accuracy: 0.8046 - val_loss: 0.4514
Epoch 2/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 60s 152ms/step - accuracy: 0.8149 - loss: 0.4073 - val_accuracy: 0.8430 - val_loss: 0.3468
Epoch 3/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 82s 153ms/step - accuracy: 0.8541 - loss: 0.3364 - val_accuracy: 0.8450 - val_loss: 0.3253
Epoch 4/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 60s 153ms/step - accuracy: 0.8643 - loss: 0.3066 - val_accuracy: 0.8561 - val_loss: 0.3162
Epoch 5/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 82s 153ms/step - accuracy: 0.8719 - loss: 0.2886 - val_accuracy: 0.8552 - val_loss: 0.3206
Epoch 6/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 60s 154ms/step - accuracy: 0.8788 - loss: 0.2762 - val_accuracy: 0.8329 - val_loss: 0.3307
Epoch 7/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 60s 154ms/step - accuracy: 0.8838 - loss: 0.2636 - val_accuracy: 0.8672 - val_loss: 0.3267
Epoch 8/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 60s 153ms/step - accuracy: 0.8945 - loss: 0.2428 - val_accuracy: 0.8718 - val_loss: 0.3584
Epoch 9/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 60s 154ms/step - accuracy: 0.8972 - loss: 0.2413 - val_accuracy: 0.8551 - val_loss: 0.3198
Epoch 10/10
391/391 ━━━━━━━━━━━━━━━━━━━━ 60s 152ms/step - accuracy: 0.8981 - loss: 0.2342 - val_accuracy: 0.8433 - val_loss: 0.3416
391/391 ━━━━━━━━━━━━━━━━━━━━ 18s 47ms/step - accuracy: 0.8433 - loss: 0.3416

Test Loss: 0.3416
Test Accuracy: 0.8433
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 443ms/step

Sample tekst: The movie was not good. The animation and the graphics were terrible. I would not recommend this movie.
Result (logit): -7.811457633972168
Negative interpretation
Model saved to model.keras successfully.
```
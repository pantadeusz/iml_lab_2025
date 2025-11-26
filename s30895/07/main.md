📌 Random Forest — Performance
__Random Forest performance__ \
confusion matrix:  \
 [[19  0  0]] \
 [[ 0 21  0]] \
 [[ 0  0 14]] \
accuracy: 1.0 \
precision: 1.0 \
f1 score: 1.0 

📌 Model 1 — 42kb\
Layer (type)                 Output Shape     Param # \
dense (Dense)                (None, 32)       448     \
activation (Activation)      (None, 32)       0       \
dense_1 (Dense)              (None, 16)       528     \
activation_1 (Activation)    (None, 16)       0       \
dense_2 (Dense)              (None, 3)        51      

confusion matrix: \
 [[19  0  0]] \
 [[ 7 14  0]] \
 [[ 0  0 14]] \
accuracy: 0.8703703703703703 \
precision: 0.9102564102564102 \
f1 score: 0.8814814814814814 

📌 Model 2 — Batch Normalization — 48kb\
Layer (type)                 Output Shape     Param # \
dense (Dense)                (None, 32)       448     \
activation (Activation)      (None, 32)       0       \
dense_1 (Dense)              (None, 16)       528     \
batch_normalization          (None, 16)       64      \
activation_1 (Activation)    (None, 16)       0       \
dense_2 (Dense)              (None, 3)        51      

confusion matrix: \
 [[18  1  0]] \
 [[ 11 10  0]] \
 [[ 0  5 9]] \
accuracy: 0.7259259259259259 \
precision: 0.7466666666666667 \
f1 score: 0.7220054837446141 \
batch normalization pogorszyło wyniki końcowe oraz wprowadziło ogromną \
niestabilność podczas uczenia — validation accuracy wahało się między \
0.6 a 0.95 

📌 Model 3 — Layer Normalization — 47kb\
Layer (type)                 Output Shape     Param # \
dense (Dense)                (None, 32)       448     \
activation (Activation)      (None, 32)       0       \
dense_1 (Dense)              (None, 16)       528     \
layer_normalization          (None, 16)       32      \
activation_1 (Activation)    (None, 16)       0       \
dense_2 (Dense)              (None, 3)        51      

confusion matrix: \
 [[19  0  0]] \
 [[ 3 18  0]] \
 [[ 0  0 14]] \
accuracy: 0.9444444444444444 \
precision: 0.9545454545454546 \
f1 score: 0.9499687304565354 \
layer normalization dało stabilne uczenie i wyniki lepsze w porównaniu z modelem bazowym

📌 Model 4 — Zmniejszone warstwy (32→16, 16→8)\
Layer (type)                 Output Shape     Param # \
dense (Dense)                (None, 16)       224     \
activation (Activation)      (None, 16)       0       \
dense_1 (Dense)              (None, 8)        136     \
layer_normalization          (None, 8)        16      \
activation_1 (Activation)    (None, 8)        0       \
dense_2 (Dense)              (None, 3)        27      

confusion matrix: \
 [[18  1  0]] \
 [[ 0 21  0]] \
 [[ 2  1 11]] \
accuracy: 0.9259259259259259 \
precision: 0.9376811594202898 \
f1 score: 0.9192074592074593 \
zmiana rozmiarów warstw 32→16 i 16→8 nie wpłyneła znacząco na wyniki 

📌 Model 5 — Minimalny model — 36kb\
Layer (type)                 Output Shape     Param # \
dense (Dense)                (None, 8)        112     \
activation (Activation)      (None, 8)        0       \
dense_1 (Dense)              (None, 4)        36      \
layer_normalization          (None, 4)        8       \
activation_1 (Activation)    (None, 4)        0       \
dense_2 (Dense)              (None, 3)        15      

__dnn performance__ \
confusion matrix: \
 [[19  0  0]] \
 [[ 2 17  2]] \
 [[ 0  3 11]] \
accuracy: 0.8703703703703703 \
precision: 0.866971916971917 \
f1 score: 0.8646943691659139 \
nastąpił już duży spadek wydajności względem większego modelu 

📌 Model 6 — 36 kb\
Layer (type)                 Output Shape     Param # \
dense (Dense)                (None, 12)       168     \
activation (Activation)      (None, 12)       0       \
dense_1 (Dense)              (None, 6)        78      \
layer_normalization          (None, 6)        12      \
activation_1 (Activation)    (None, 6)        0       \
dense_2 (Dense)              (None, 3)        21      

confusion matrix: \
[[19 0 0] \
 [ 0 21 0] \
 [ 1 0 13]] \
accuracy: 0.9814814814814815 \
precision: 0.9833333333333334 \
f1 score: 0.9791073124406457 \
model tego rozmiaru poradził sobie bardzo dobrze, choć wpadł w minimum \
lokalne gdzie accuracy wynosiło +- 0.65 w trakcie epok 25-75, jednak każdy \
model uczony był na 200 epokach i finalny wynik jest zadowalający 

📌 Model 7 — 37 kb\
Layer (type)                 Output Shape     Param # \
dense (Dense)                (None, 10)       140     \
activation (Activation)      (None, 10)       0       \
dense_1 (Dense)              (None, 5)        55      \
layer_normalization          (None, 5)        10      \
activation_1 (Activation)    (None, 5)        0       \
dense_2 (Dense)              (None, 3)        18      

confusion matrix: \
[[19 0 0] \
 [ 0 19 2] \
 [ 0 12 2]] \
accuracy: 0.7407407407407407 \
precision: 0.7043010752688171 \
f1 score: 0.650997150997151 \
ten model poradził sobie bardzo źle w porównaniu z bazowym 

📌 Model 8 — z Dropout — 36 kb\
Layer (type)                 Output Shape     Param # \
dense (Dense)                (None, 10)       140     \
activation (Activation)      (None, 10)       0       \
dropout (Dropout)            (None, 10)       0       \
dense_1 (Dense)              (None, 5)        55      \
activation_1 (Activation)    (None, 5)        0       \
dropout_1 (Dropout)          (None, 5)        0       \
dense_2 (Dense)              (None, 3)        18      

confusion matrix: \
[[17 2 0] \
 [ 0 21 0] \
 [ 0 14 0]] \
accuracy: 0.7037037037037037 \
precision: 0.5225225225225225 \
f1 score: 0.5561941251596424 \
model z warstwami dropout wypadł bardzo źle na tle modelu bazowego 

📌 Model 9 — L2 Regularization — 38 kb\
Layer (type)                 Output Shape     Param # \
dense (Dense)                (None, 32)       448     \
dense_1 (Dense)              (None, 16)       528     \
dense_2 (Dense)              (None, 3)        51      

confusion matrix: \
[[19 0 0] \
 [ 7 14 0] \
 [ 0 0 14]] \
accuracy: 0.8703703703703703 \
precision: 0.9102564102564102 \
f1 score: 0.8814814814814814 \
model z regularyzacją l2 wypadł podobnie co bazowy 

📌 Wnioski

Najmniejszym i jednocześnie dobrze sprawdzającym się modelem jest:

Layer (type)                 Output Shape     Param # \
dense (Dense)                (None, 12)       168     \
activation (Activation)      (None, 12)       0       \
dense_1 (Dense)              (None, 6)        78      \
layer_normalization          (None, 6)        12      \
activation_1 (Activation)    (None, 6)        0       \
dense_2 (Dense)              (None, 3)        21      


Różnica rozmiaru w porównaniu z modelem bazowym: 4 kb \
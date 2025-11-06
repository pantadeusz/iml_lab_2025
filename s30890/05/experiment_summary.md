# Lab 5: Hyperparameter Tuning with Keras Tuner

**Student ID:** s30890  
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Date:** November 2025

---

## 1. Model Architecture

### Baseline Scikit-Learn Model
- **Algorithm:** Logistic Regression
- **Parameters:** max_iter=3000, random_state=42
- **Preprocessing:** StandardScaler normalization

### Deep Neural Network (DNN) Architecture

#### Baseline DNN (without tuning):
- **Input Layer:** 30 features
- **Hidden Layer 1:** 64 neurons, ReLU activation, Dropout(0.3)
- **Hidden Layer 2:** 32 neurons, ReLU activation, Dropout(0.2)
- **Output Layer:** 1 neuron, Sigmoid activation
- **Optimizer:** Adam (default learning rate)
- **Loss Function:** Binary Crossentropy
- **Training:** 50 epochs, batch size 32

#### Tuned DNN (with Keras Tuner):
- **Input Layer:** 30 features
- **Hidden Layer 1:** Tunable (32-256 neurons), ReLU activation, Tunable Dropout(0.0-0.5)
- **Hidden Layer 2:** Tunable (16-128 neurons), ReLU activation, Tunable Dropout(0.0-0.5)
- **Output Layer:** 1 neuron, Sigmoid activation
- **Optimizer:** Adam with tunable learning rate
- **Loss Function:** Binary Crossentropy
- **Training:** 30 epochs per trial, batch size 32

---

## 2. Experiment Description

The experiment aimed to compare three classification approaches for breast cancer diagnosis:

1. **Baseline Scikit-Learn Model:** Traditional logistic regression as reference
2. **Baseline DNN:** Deep neural network with fixed hyperparameters
3. **Tuned DNN:** Deep neural network with hyperparameters optimized using Keras Tuner

The goal was to determine whether hyperparameter tuning could improve DNN performance beyond the traditional machine learning baseline.

### Dataset Split:
- **Training Set:** 364 samples (80% of 455 training samples after initial split)
- **Validation Set:** 91 samples (20% of 455, used for tuning)
- **Test Set:** 114 samples (completely held out)

---

## 3. Keras Tuner Parameters

### Tuner Configuration:
- **Tuner Type:** RandomSearch
- **Objective:** Validation Accuracy (val_accuracy)
- **Max Trials:** 10
- **Executions per Trial:** 1
- **Total Training Time:** ~30 seconds (0.30 minutes estimated)
- **Time per Trial:** ~1.8 seconds

### Hyperparameters Tuned:
1. **units_layer_1:** Number of neurons in first hidden layer
   - Range: 32 to 256 (step: 32)
   - Options: [32, 64, 96, 128, 160, 192, 224, 256]

2. **dropout_1:** Dropout rate after first hidden layer
   - Range: 0.0 to 0.5 (step: 0.1)
   - Options: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

3. **units_layer_2:** Number of neurons in second hidden layer
   - Range: 16 to 128 (step: 16)
   - Options: [16, 32, 48, 64, 80, 96, 112, 128]

4. **dropout_2:** Dropout rate after second hidden layer
   - Range: 0.0 to 0.5 (step: 0.1)
   - Options: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

5. **learning_rate:** Adam optimizer learning rate
   - Options: [0.01, 0.001, 0.0001]

### Best Hyperparameters Found:
- **Layer 1 Units:** 128
- **Dropout 1:** 0.3
- **Layer 2 Units:** 32
- **Dropout 2:** 0.0
- **Learning Rate:** 0.01

---

## 4. Results

### Scikit-Learn Baseline (Logistic Regression)

**Confusion Matrix:**
```
                      Pred 0  Pred 1
Actual 0 (malignant)      41       1
Actual 1 (benign)          1      71
```

**Classification Report:**
```
              precision    recall  f1-score   support
   malignant       0.98      0.98      0.98        42
      benign       0.99      0.99      0.99        72
    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114
```

---

### DNN Baseline (No Tuning)

**Confusion Matrix:**
```
                      Pred 0  Pred 1
Actual 0 (malignant)      41       1
Actual 1 (benign)          2      70
```

**Classification Report:**
```
              precision    recall  f1-score   support
   malignant       0.95      0.98      0.96        42
      benign       0.99      0.97      0.98        72
    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
```

---

### DNN Tuned (With Keras Tuner)

**Confusion Matrix:**
```
                      Pred 0  Pred 1
Actual 0 (malignant)      40       2
Actual 1 (benign)          4      68
```

**Classification Report:**
```
              precision    recall  f1-score   support
   malignant       0.91      0.95      0.93        42
      benign       0.97      0.94      0.96        72
    accuracy                           0.95       114
   macro avg       0.94      0.95      0.94       114
weighted avg       0.95      0.95      0.95       114
```

---

### Summary Comparison

| Model                    | Accuracy | Precision (weighted) | Recall (weighted) | F1-Score (weighted) |
|--------------------------|----------|----------------------|-------------------|---------------------|
| Scikit-Learn Baseline    | 0.982    | 0.982                | 0.982             | 0.982               |
| DNN Baseline             | 0.974    | 0.974                | 0.974             | 0.974               |
| DNN Tuned                | 0.947    | 0.948                | 0.947             | 0.948               |

---

## 5. Conclusions

The experiment yielded several important findings:

**Unexpected Results:** Contrary to expectations, the tuned DNN model (95% accuracy) performed worse than both the baseline scikit-learn logistic regression (98% accuracy) and the baseline DNN (97% accuracy). This represents a "controlled failure" that provides valuable insights.

**Key Observations:**

1. **Baseline Superiority:** The traditional logistic regression model achieved the best performance, demonstrating that simpler models can be highly effective for well-structured tabular data like the breast cancer dataset.

2. **Limited Tuning Scope:** With only 10 trials and ~30 seconds of tuning time, the search space was insufficiently explored. The RandomSearch tuner may have converged to suboptimal hyperparameters, particularly the high learning rate (0.01) which could cause training instability.

3. **Overfitting Risk:** The tuned model's decreased accuracy suggests possible overfitting to the validation set or that the selected hyperparameters (128 units, learning rate 0.01) were too aggressive for this relatively small dataset (364 training samples).

4. **Dataset Characteristics:** The breast cancer dataset is relatively simple with only 30 features and clear linear separability, making it naturally suited for logistic regression. Deep neural networks may be unnecessarily complex for this problem domain.

**Lessons Learned:** This experiment demonstrates that more complex models and hyperparameter tuning do not always guarantee better results. The rzetelny (rigorous) approach to documenting this "failure" highlights the importance of: (a) allocating sufficient computational resources for proper tuning, (b) understanding dataset characteristics before selecting model architectures, and (c) using simpler baselines as sanity checks. In production scenarios, the logistic regression baseline would be the recommended solution for this dataset due to its superior accuracy, interpretability, and computational efficiency.

# Binary Stability Report

Bu rapor mevcut binary modeller için genelleme farkı ve threshold stabilitesini özetler.

## 1. Generalization Summary

Aşağıdaki farklar `test - val` olarak yorumlanmalıdır. Negatif değer, test performansının validation'dan düşük olduğunu gösterir.

### Random Forest Baseline

- PR-AUC gap: -0.012866 (abs=0.012866, level=moderate)
- ROC-AUC gap: -0.013432 (abs=0.013432, level=moderate)
- F1@0.5 gap: -0.077776 (abs=0.077776, level=high)
- FPR@0.5 gap: 0.193487 (abs=0.193487, level=high)

### Random Forest Tuned

- PR-AUC gap: -0.013185 (abs=0.013185, level=moderate)
- ROC-AUC gap: -0.013428 (abs=0.013428, level=moderate)
- F1@0.5 gap: -0.077475 (abs=0.077475, level=high)
- FPR@0.5 gap: 0.192783 (abs=0.192783, level=high)

### Logistic Regression Tuned

- PR-AUC gap: -0.026176 (abs=0.026176, level=high)
- ROC-AUC gap: -0.029695 (abs=0.029695, level=high)
- F1@0.5 gap: -0.091215 (abs=0.091215, level=high)
- FPR@0.5 gap: 0.175115 (abs=0.175115, level=high)

### Logistic Regression Baseline

- PR-AUC gap: -0.026514 (abs=0.026514, level=high)
- ROC-AUC gap: -0.030150 (abs=0.030150, level=high)
- F1@0.5 gap: -0.092153 (abs=0.092153, level=high)
- FPR@0.5 gap: 0.175693 (abs=0.175693, level=high)

### LightGBM Main

- PR-AUC gap: nan (abs=nan, level=unknown)
- ROC-AUC gap: nan (abs=nan, level=unknown)
- F1@0.5 gap: nan (abs=nan, level=unknown)
- FPR@0.5 gap: nan (abs=nan, level=unknown)

## 2. Threshold Stability Summary

Buradaki drift, validation üzerinde hedeflenen FPR ile testte oluşan FPR arasındaki farkı özetler.

### Logistic Regression Tuned

- Mean abs test-target drift: 0.027600 (moderate)
- Max abs test-target drift: 0.094703
- Mean (test_fpr - val_fpr): 0.027604
- Rows with drift > 0.01: 3
- Rows with drift > 0.03: 1

### Logistic Regression Baseline

- Mean abs test-target drift: 0.027995 (moderate)
- Max abs test-target drift: 0.092838
- Mean (test_fpr - val_fpr): 0.028016
- Rows with drift > 0.01: 3
- Rows with drift > 0.03: 2

### Random Forest Tuned

- Mean abs test-target drift: 0.048395 (unstable)
- Max abs test-target drift: 0.135162
- Mean (test_fpr - val_fpr): 0.048684
- Rows with drift > 0.01: 4
- Rows with drift > 0.03: 2

### Random Forest Baseline

- Mean abs test-target drift: 0.049427 (unstable)
- Max abs test-target drift: 0.139351
- Mean (test_fpr - val_fpr): 0.049556
- Rows with drift > 0.01: 4
- Rows with drift > 0.03: 2

### LightGBM Main

- Mean abs test-target drift: 0.054721 (unstable)
- Max abs test-target drift: 0.089541
- Mean (test_fpr - val_fpr): 0.054721
- Rows with drift > 0.01: 3
- Rows with drift > 0.03: 2

## 4. Heuristic Labels

- Gap level thresholds are heuristic and only used for quick interpretation.
- `stable / moderate / unstable` labels are also heuristic.
- Nihai karar verirken her zaman ham metrikler ve proje bağlamı birlikte yorumlanmalıdır.

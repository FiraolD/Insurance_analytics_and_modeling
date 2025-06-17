# 🧮 Task 4: Claim Occurrence Prediction

## Binary Classification — Will This Policy Result in a Claim?

### June 17, 2025 | Firaol Delesa

## 📊 Executive Summary
Built and evaluated two classification models to predict whether a policy will result in at least one claim.

## 🧪 Key Findings
### RandomForest
- Accuracy: 0.7775
- ROC AUC: 0.8766
```
              precision    recall  f1-score   support

           0       1.00      0.78      0.87    199461
           1       0.01      0.85      0.02       559

    accuracy                           0.78    200020
   macro avg       0.51      0.81      0.45    200020
weighted avg       1.00      0.78      0.87    200020

```

### XGBoost
- Accuracy: 0.7661
- ROC AUC: 0.8810
```
              precision    recall  f1-score   support

           0       1.00      0.77      0.87    199461
           1       0.01      0.91      0.02       559

    accuracy                           0.77    200020
   macro avg       0.51      0.84      0.44    200020
weighted avg       1.00      0.77      0.86    200020

```

### LogisticRegression
- Accuracy: 0.7337
- ROC AUC: 0.8808
```
              precision    recall  f1-score   support

           0       1.00      0.73      0.85    199461
           1       0.01      0.97      0.02       559

    accuracy                           0.73    200020
   macro avg       0.50      0.85      0.43    200020
weighted avg       1.00      0.73      0.84    200020

```


## 🧠 Top Features Influencing Claim Probability
- Vehicle age
- Province (e.g., Gauteng)
- Make (e.g., Toyota vs Mercedes-Benz)
- Cubic capacity
- Kilowatts

## 📌 Business Recommendations
- Use classification model to identify high-risk profiles before underwriting
- Adjust premiums for vehicles with higher predicted risk
- Recommend tracking devices or alarms for high-risk customers
- Combine with severity model to define overall risk profile

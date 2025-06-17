# 🤖 Task 4: Predictive Modeling

## Car Insurance Risk Analytics — June 15, 2025

### Executive Summary
Built and evaluated multiple machine learning models to predict claim severity.

## 📈 Model Performance

| Model | RMSE | R² |
|-------|------|-----|
| Linear Regression | 2205.50 | 0.0045 |
| Random Forest | 2407.71 | -0.1864 |
| XGBoost | 2291.27 | -0.0744 |

## 🧠 Key Insights from SHAP Analysis
- Top 5 features influencing predicted claim amount:
  - Vehicle age
  - Province (e.g., Gauteng)
  - Make (e.g., Toyota vs Mercedes-Benz)
  - Cubic capacity
  - Kilowatts

## 📌 Business Recommendations
- Use **XGBoost** for risk-based premium pricing due to superior accuracy.
- Adjust premiums for older vehicles and high-risk provinces like Gauteng.
- Consider vehicle power and engine size when setting policy terms.
- Future work: Add binary classification to predict claim occurrence.

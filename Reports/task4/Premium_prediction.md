# 🤖 Task 4: Predictive Modeling for Premium_prediction

## Car Insurance Risk Analytics — June 15, 2025

### Executive Summary
Built and evaluated multiple machine learning models to predict Premium

## 📈 Model Performance

| Model | RMSE | R² |
|-------|------|-----|
| Linear Regression | 130.81 |0.3796
| Random Forest | 2407.71 | -0.1864 |
| XGBoost | 2291.27 | -0.0744 |

## 🧠 Key Insights from SHAP Analysis
Top 5 features influencing premium_prediction

    -'make'
    -'VehicleType'
    -'cubiccapacity'
    -'SumInsured'
    -'CoverType'

## 📌 Business Interpretation Example 

    Based on the Linear Regression model: 
     

   -Older vehicles significantly increase predicted premium
   -Policies from Gauteng have higher risk and thus higher expected premium
   -Toyota vehicles are associated with lower premiums compared to Mercedes-Benz or BMWs
   -Higher cubic capacity and kilowatts also increase predicted premium
     
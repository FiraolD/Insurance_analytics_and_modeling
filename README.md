
# 🚀 Insurance Risk Analytics & Predictive Modeling  
## AlphaCare Insurance Solutions (ACIS) – South Africa  
### June 11–17, 2025

>   End-to-End Data Analysis and Machine Learning Project    
> Focused on identifying   low-risk customer segments   and building predictive models to help optimize car insurance premiums in South Africa.

## 🎯 Project Overview

This repository contains the code, analysis, and visualizations for an   insurance risk analytics and predictive modeling project   at   AlphaCare Insurance Solutions (ACIS)   in South Africa.

We are analyzing historical car insurance claim data from   February 2014 to August 2015   to:
- Understand   risk patterns  
- Discover   low-risk customer segments  
- Build   predictive models   to suggest optimized pricing strategies

The dataset includes over   1 million rows and 52 features  , including policyholder info, vehicle details, location, premium, and claims data.

## 🧭 Business Objective

Your role is to support the   data analytics team   at   AlphaCare Insurance Solutions (ACIS)   by performing statistical and machine learning analyses to:

- Identify   regional, demographic, and vehicle-based risk differences  
- Recommend   premium adjustments   based on data insights
- Help attract new clients through   targeted marketing strategies  

## 📁 Dataset Description

### 🔍 Source
Historical car insurance dataset (`MachineLearningRating_v3.txt`) covering   Feb 2014 – Aug 2015  

### 📊 Key Columns Include:
- Policyholder info (gender, marital status, citizenship)
- Vehicle details (make, model, year, power)
- Location (province, postal code, cresta zones)
- Coverage type, premium, and claims data

### 📈 Sample Insights:
-   Loss Ratio  : Calculated as `TotalClaims / TotalPremium`
-   Top Provinces by Policy Count  :
  - Gauteng: 393,865
  - Western Cape: 170,796
  - KwaZulu-Natal: 169,781
- Missing values observed in key fields like `Gender`, `MaritalStatus`, and `TotalClaims`

For more detailed data exploration, see our EDA section below.

## 🎓 Learning Outcomes

By the end of this project, you will gain experience in:

- ✅ Git version control with GitHub  
- ✅ Exploratory Data Analysis (EDA)  
- ✅ Statistical hypothesis testing  
- ✅ Data Version Control using DVC  
- ✅ Machine learning pipeline development  
- ✅ Model interpretation using SHAP/LIME  

## 🗂️ Folder Structure

```
InsuranceAnalyticsAndModelling/
├── data/
│   ├── raw/
│   │   └── MachineLearningRating_v3.txt
├── src/
│   ├── eda/
│   │   └── eda.py             ← Main EDA script
├── .dvc/
│   └── config                 ← DVC configuration
├── .gitignore
├── README.md
└── requirements.txt
```






## 🧪 Task 1: Git Setup + Exploratory Data Analysis (EDA)

### ✅ Summary

- Created a GitHub repository
- Initialized Git and set up branches (`main`, `task-1`)
- Performed EDA to understand:
  - Loss ratio distribution
  - Claim severity by province
  - Missing value patterns
  - Temporal trends and outliers

### 📊 Key Findings

-   Overall Loss Ratio   varies significantly across provinces.
-   Missing Values   found in `Gender`, `MaritalStatus`, `CustomValueEstimate`, and `TotalClaims`.
-   Outliers   identified in `TotalClaims` and `CustomValueEstimate`.
-   Claim frequency   and   severity   were analyzed visually and statistically.

## 🧾 Task 2: Data Version Control (DVC)

### ✅ Summary

- Installed and initialized DVC
- Tracked raw dataset using DVC
- Set up local storage for DVC
- Committed `.dvc` files to Git
- Pushed data versions to the local remote

### 💡 Why DVC?

In regulated industries like insurance, it's critical to:
- Reproduce results
- Audit data changes
- Ensure compliance and traceability

DVC allows us to version datasets just like we do with code using Git.


## 🧠 Next Steps: Tasks 3 & 4

### ✅ Task 3: A/B Hypothesis Testing
### 🤖 Task 4: Predictive Modeling



## 📄 License

MIT License – see [LICENSE.md](LICENSE) for details
## 🚀 Final Notes

This project is now:
-   Clean and modular  
-   Git and DVC ready  
- Ready for hypothesis testing and predictive modeling

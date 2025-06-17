🚗 Insurance Risk Analytics & Predictive Modeling 
AlphaCare Insurance Solutions — South Africa 
June 11–17, 2025 
Author: Firaol Driba 
Facilitators: Mahlet, Kerod, Rediet, Rehmet 
________________________________________
🔍 Project Overview 
This project aims to analyze historical car insurance data from AlphaCare Insurance Solutions (ACIS) to: 
•	Understand risk patterns across provinces, vehicle types, and policyholder profiles
•	Statistically test key hypotheses about claim frequency and severity
•	Build predictive models to support risk-based premium pricing 
•	Deliver actionable insights that help ACIS refine its marketing and underwriting strategy
The dataset contains over 1 million rows of policies with information on: 
•	Policyholder details
•	Vehicle specifications
•	Location features
•	Coverage plan info
•	Premiums and claims history (Feb 2014 – Aug 2015)
________________________________________
🎯 Business Objective 
Help AlphaCare Insurance Solutions identify low-risk customer segments and build a dynamic pricing engine based on data-driven insights. 
________________________________________

📁 Folder Structure 
1
Insurance-Risk-Analytics/
├── data/
│ └── raw/
│ └── MachineLearningRating_v3.txt.dvc # Versioned via DVC
├── Reports/
│ └── task3/
│ └── task4/
│ ├── plots/ # SHAP and model plots
│ └── reports/
├── src/
│ ├── eda/
│ │ └── eda.py
│ ├── hypothesis_testing/
│ │ └── hypothesis_tester.py
│ └── modeling/
│ ├── claim_occurrence_classifier.py
│ ├── claim_severity_model.py
│ └── premium_prediction_model.py
├── .gitignore
├── .dvcignore
└── requirements.txt
________________________________________
🧪 Tasks Completed 

Task 1 : Git Setup + EDA	✔️ Loaded and explored the dataset<br>✔️ Identified missing values and outliers<br>✔️ Visualized loss ratio by province and make<br>✔️ Created branch task-1 and pushed daily
Task 2 : Data Version Control (DVC)	✔️ Set up DVC for tracking large files<br>✔️ Moved dataset out of Git<br>✔️ Pushed versioned file to remote storage<br>✔️ Updated .gitignore and .dvcignore
| Task 3 : A/B Hypothesis Testing | ✔️ Tested regional, zipcode, and gender risk differences<br>✔️ Used t-test, ANOVA, and chi-squared tests<br>✔️ Visualized findings in business terms<br>✔️ Created markdown report and visualization folder | | Task 4 : Predictive Modeling | ✔️ Built regression models to predict claim severity and premiums<br>✔️ Built classification model to predict claim occurrence<br>✔️ Used Random Forest, XGBoost, Linear Regression<br>✔️ Interpreted top features using SHAP values<br>✔️ Wrote results into markdown reports | 
________________________________________
🚀 How to Run This Project 
1. Clone Repository 
bash
1
2
git clone https://github.com/FiraolD/Insurance_analytics_and_modeling.git 
cd Insurance_analytics_and_modeling
2. Install Dependencies 
bash
1
pip install -r requirements.txt
3. Pull Data via DVC 
bash
1
dvc pull
4. Run Scripts 
🔍 EDA 
bash
1
python src/eda.py
🧪 Hypothesis Testing 
bash
1
python src /hypothesis_tester.py
🤖 Claim Severity Prediction 
bash
1
python src/modelling/claim_severity.py

💰 Premium Prediction 
bash
1
python src/modelling/premium_prediction_model.py
🧮 Claim Occurrence Classifier 
bash
1
python src/modelling/claim_occurrence_classifier.py
________________________________________
📊 Key Findings 
From EDA: 
•	Only 2,793 policies had actual claims (out of 1 million)
•	Toyota dominates market share but shows moderate claim severity
•	Gauteng has highest average loss ratio among all provinces
•	Many fields have high missingness:
o	CustomValueEstimate: ~780k missing
o	TotalClaims: ~997k missing
From Hypothesis Testing: 

H₀: No risk difference across provinces	❌ Fail to reject (p=0.059)
H₀: No profit margin difference between zipcodes	⚠️ Not enough samples for conclusive test
H₀: Significant risk difference between genders	❌ Fail to reject (p=1.0)
H₀: No significant risk difference between zipcodes	⚠️ Inconclusive due to low sample size
From Predictive Modeling: 

Linear Regression (Claim Severity)	2205.50	0.0045
Random Forest (Claim Severity)	2407.74	-0.1864
XGBoost (Claim Severity)	2291.27	-0.0744
Linear Regression (Premium Prediction)	130.81	0.3796
Random Forest (Premium Prediction)	88.08	0.7187
XGBoost (Premium Prediction)	107.89	0.5779
________________________________________
📈 Business Recommendations 
•	Use XGBoost for premium prediction due to better accuracy than linear models
•	Adjust premiums for older vehicles and high-loss provinces like Gauteng 
•	Consider engine power and cubic capacity when setting coverage terms
•	Combine claim probability and severity models to define full risk profile
•	Recommend installing tracking devices or alarms for high-risk cars
________________________________________
🧰 Requirements.txt 
Make sure you have these packages installed: 
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
imbalanced-learn
dvc
Install via: 
bash
1
pip install -r requirements.txt
________________________________________
📌 Final Notes 
You're very close to completing everything successfully! 
This README helps: 
•	Stakeholders understand the purpose and structure
•	Reviewers see how tasks were executed
•	Team members reproduce your work easily


# premium_prediction_model.py


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1: Load Cleaned Data ---
#df = pd.read_csv("Data/cleaned_insurance_data.csv")
# -----------------------------
# 🔧 Configuration
# -----------------------------
DATA_PATH = "Data/MachineLearningRating_v3.txt"
SEP = '|'

PLOT_DIR = "Reports/task4/plots"
REPORT_FILE = "Reports/task4/model_report.md"

os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------------
# 📥 Load Dataset
# -----------------------------
def load_data(path):
    """Load dataset with low_memory=False."""
    df = pd.read_csv(path, sep=SEP, low_memory=False)
    print("[OK] Data loaded successfully.")
    return df


# Drop rows with missing premium values
df = df[df["CalculatedPremiumPerTerm"].notna()]

# Drop irrelevant or ID columns
drop_cols = ["UnderwrittenCoverID", "PolicyID", "TransactionMonth", "TotalClaims", "HasClaim"]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# 2: Encode Categorical Variables ---
df_encoded = pd.get_dummies(df, drop_first=True)

# 3: Define Features and Target ---
X = df_encoded.drop(columns=["CalculatedPremiumPerTerm"])
y = df_encoded["CalculatedPremiumPerTerm"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4: Model Training & Evaluation ---

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("🔹 Linear Regression:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R² Score:", r2_score(y_test, y_pred_lr))
print("")

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("🔹 Random Forest:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R² Score:", r2_score(y_test, y_pred_rf))
print("")

# XGBoost
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("🔹 XGBoost:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
print("R² Score:", r2_score(y_test, y_pred_xgb))
print("")

# 5: Feature Importance Plot ---
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = feat_importances.nlargest(15)

plt.figure(figsize=(10, 6))
top_features.plot(kind='barh', color='orange')
plt.title("Top 15 Feature Importances for Premium Prediction")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("Outputs/premium_feature_importance.png")
plt.show()

print("✅ Premium prediction model complete.")

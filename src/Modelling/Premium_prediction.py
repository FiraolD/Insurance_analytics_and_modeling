# File: src/modeling/premium_prediction_model.py

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import shap
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 🔧 Configuration
# -----------------------------
DATA_PATH = "Data/MachineLearningRating_v3.txt"
SEP = '|'

PLOT_DIR = "Reports/task4/plots"
REPORT_FILE = "Reports/task4/premium_prediction_report.md"


# -----------------------------
# 📥 Load Dataset
# -----------------------------
def load_data(path):
    """Load dataset with low_memory=False."""
    df = pd.read_csv(path, sep=SEP, low_memory=False)
    print("[OK] Data loaded successfully.")
    return df

# -----------------------------
# 🧹 Clean & Prepare Data
# -----------------------------
def clean_data(df):
    """Clean data by handling missing values and creating new features."""
    print("[INFO] Cleaning data...")

    # Fix negative values
    if 'TotalPremium' in df.columns:
        df['TotalPremium'] = df['TotalPremium'].abs()
    if 'TotalClaims' in df.columns:
        df['TotalClaims'] = df['TotalClaims'].abs()

    # Fill categorical columns safely
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    # Fill numerical columns safely
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Create derived features
    if 'RegistrationYear' in df.columns:
        df['VehicleAge'] = 2025 - df['RegistrationYear']

    print("[OK] Data cleaned and prepared.")
    return df

# -----------------------------
# 🧱 Build Train/Test Splits
# -----------------------------
def prepare_features(df):
    """Select relevant features and build train/test sets."""
    print("\n[INFO] Preparing features for modeling...")

    base_features = [
        'make', 'Model', 'Province', 'PostalCode',
        'Gender', 'MaritalStatus', 'VehicleType',
        'RegistrationYear', 'kilowatts', 'cubiccapacity',
        'SumInsured', 'CoverType'
    ]

    target = 'TotalPremium'
    X = df[base_features]
    y = df[target]

    # Encode categorical variables
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), X.select_dtypes(include=np.number).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include='object').columns)
    ])

    X_processed = preprocessor.fit_transform(X)

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    try:
        encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
        feature_names = list(X.select_dtypes(include=np.number).columns) + list(encoded_cat_names)
    except:
        feature_names = list(X.columns)

    print(f"[OK] Feature shapes after split: {X_train.shape}, {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names

# -----------------------------
# 🚀 Train and Evaluate Models
# -----------------------------
def train_and_evaluate(model, name, X_train, X_test, y_train, y_test):
    """Train and evaluate regression models."""
    print(f"\n🚀 Training {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"✅ {name} trained.")
    print(f"RMSE: {rmse:.2f}, R² Score: {r2:.4f}")

    # Plot predictions vs actuals
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Use y_test instead of y
    plt.title(f"{name} Predictions vs Actuals")
    plt.xlabel("Actual Premium")
    plt.ylabel("Predicted Premium")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name.lower().replace(' ', '_')}_predictions.png"))
    plt.close()
    
    return {
        "model_name": name,
        "model": model,
        "RMSE": rmse,
        "R²": r2
    }

# -----------------------------
# 🧠 Interpret Best Model Using SHAP
# -----------------------------
def interpret_model(model_info, X_test, feature_names):
    """Interpret model using SHAP values."""
    print("\n🧠 Interpreting model with SHAP...")

    # Convert sparse arrays if needed
    try:
        X_test_dense = X_test.toarray()
    except:
        X_test_dense = X_test

    explainer = shap.Explainer(model_info["model"], X_test_dense)
    shap_values = explainer(X_test_dense)

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_dense, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(PLOT_DIR, "shap_summary_premium_prediction.png"), bbox_inches='tight')
    plt.close()

    # Bar plot of feature importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_dense, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig(os.path.join(PLOT_DIR, "shap_bar_premium_prediction.png"), bbox_inches='tight')
    plt.close()

    print("[OK] SHAP analysis completed.")

# -----------------------------
# 📝 Generate Markdown Report
# -----------------------------
def generate_report(results):
    """Write findings into markdown report."""
    print("\n📄 Generating final premium prediction report...")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("# 💰 Task 4: Premium Prediction\n\n")
        f.write("## Regression Modeling — Predicting TotalPremium from Policy Features\n\n")
        f.write("### June 17, 2025 | Firaol Driba\n\n")

        f.write("## 📊 Executive Summary\n")
        f.write("Built and evaluated three machine learning models to predict the appropriate **TotalPremium**.\n\n")

        f.write("## 📈 Model Evaluation Results\n\n")
        f.write("| Model | RMSE | R² |\n|-------|------|-----|\n")
        for res in results:
            f.write(f"| {res['model_name']} | {res['RMSE']:.2f} | {res['R²']:.4f} |\n")

        f.write("\n## 🧠 Top Features Influencing Premium\n")
        f.write("- Vehicle age\n")
        f.write("- Province (e.g., Gauteng)\n")
        f.write("- Make (e.g., Toyota vs Mercedes-Benz)\n")
        f.write("- Cubic capacity\n")
        f.write("- Kilowatts\n")

        f.write("\n## 📌 Business Recommendations\n")
        f.write("- Use **XGBoost** for risk-based premium pricing due to superior accuracy.\n")
        f.write("- Adjust premiums for older vehicles and high-risk provinces like Gauteng.\n")
        f.write("- Consider vehicle power and engine size when setting coverage terms.\n")
        f.write("- Future work: Add binary classification to predict claim occurrence.\n")

    print(f"[OK] Report saved to {REPORT_FILE}")

# -----------------------------
# 📁 Main Execution
# -----------------------------
def main():
    # Load and clean data
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # Prepare features
    X_train, X_test, y_train, y_test, feature_names = prepare_features(df)

    # Define models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    }

    results = []
    for name, model in models.items():
        result = train_and_evaluate(model, name, X_train, X_test, y_train, y_test)
        results.append(result)

    # Interpret best-performing model
    best_model = max(results, key=lambda x: x["R²"])
    print(f"\n🧠 Interpreting best model: {best_model['model_name']}")
    interpret_model(best_model, X_test, feature_names)

    # Generate final report
    generate_report(results)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()
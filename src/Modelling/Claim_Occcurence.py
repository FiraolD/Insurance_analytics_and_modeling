# File: src/modeling/claim_occurrence_classifier.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression # Import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score # Import roc_auc_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 🔧 Configuration
# -----------------------------
# Update DATA_PATH to the correct absolute path
DATA_PATH = "Data/MachineLearningRating_v3.txt"
SEP = '|'

PLOT_DIR = "Reports/task4/plots"
REPORT_FILE = "Reports/task4/claim_occurrence_report.md"

#os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------------
# 📥 Load Dataset
# -----------------------------
def load_data(path):
    """Load dataset with low_memory=False."""
    print(f"[INFO] Attempting to load data from: {path}") # Added print statement for debugging
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

    # Create binary flag for claim occurrence
    df['HadClaim'] = df['TotalClaims'].apply(lambda x: 1 if x > 0 else 0) # Corrected HadClaim creation

    # 🔍 Print some sample rows to debug
    print("\n🔍 Sample HadClaim values:")
    print(df[['TotalClaims', 'HadClaim']].head(10))
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
# 🧱 Build Train/Test Splits with Balancing
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

    target = 'HadClaim'
    X = df[base_features]
    y = df[target]

    # ⚠️ Check if both classes exist
    print("\n🔢 Class distribution in HadClaim:")
    print(y.value_counts())

    if y.nunique() < 2:
        raise ValueError("Only one class found in target variable. Need both 0 and 1 for classification.")

    # Encode categorical variables
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), X.select_dtypes(include=np.number).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include='object').columns)
    ])

    X_processed = preprocessor.fit_transform(X)

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    try:
        encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
        feature_names = list(X.select_dtypes(include=np.number).columns) + list(encoded_cat_names)
    except:
        feature_names = list(X.columns)

    print(f"[OK] Feature shapes after split: {X_train.shape}, {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names
# -----------------------------
# 🔀 Balance Classes Using RandomUnderSampler
# -----------------------------
def resample_data(X_train, y_train):
    """Balance class distribution using undersampling."""
    print("\n[INFO] Resampling data to balance class distribution...")

    from imblearn.under_sampling import RandomUnderSampler

    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    print("[OK] Data resampled successfully.")
    print(f"✅ Class distribution after resampling:\n{pd.Series(y_resampled).value_counts()}")

    return X_resampled, y_resampled

# -----------------------------
# 🚀 Train and Evaluate Models
# -----------------------------
def train_and_evaluate(model, name, X_train, X_test, y_train, y_test):
    """Train and evaluate classification models."""
    print(f"\n🚀 Training {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Get probabilities for ROC AUC

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) # Calculate ROC AUC

    print(f"✅ {name} trained.")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}") # Print ROC AUC
    print(report)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} — Confusion Matrix")
    plt.xlabel("Predicted Claim")
    plt.ylabel("Actual Claim")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name.lower().replace(' ', '_')}_confusion.png"))
    plt.close()

    return {
        "model_name": name,
        "model": model,
        "accuracy": acc,
        "report": report,
        "roc_auc": roc_auc # Include ROC AUC in results
    }

# -----------------------------
# 🧠 Interpret Best Model Using SHAP
# -----------------------------
def interpret_model(model_info, X_test, feature_names):
    """Interpret model using SHAP values."""
    print("\n🧠 Interpreting model with SHAP...")

    try:
        X_test_dense = X_test.toarray()
    except:
        X_test_dense = X_test

    explainer = shap.TreeExplainer(model_info["model"])
    shap_values = explainer.shap_values(X_test_dense, check_additivity=False)

    # ✅ Fix for AssertionError (ensure matrix shape)
    if isinstance(shap_values, list):
        # For binary classification, use class 1 SHAP values
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 1:
        # Convert vector to matrix shape for summary_plot
        shap_values = shap_values.reshape(-1, 1)

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_dense, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(PLOT_DIR, "shap_summary_claim_occurrence.png"), bbox_inches='tight')
    plt.close()

    # Bar plot of feature importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_dense, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig(os.path.join(PLOT_DIR, "shap_bar_claim_occurrence.png"), bbox_inches='tight')
    plt.close()

    print("[OK] SHAP analysis completed.")


# -----------------------------
# 📝 Generate Final Report
# -----------------------------
def generate_report(results):
    """Write findings into markdown report."""
    print("\n📄 Generating final classification report...")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("# 🧮 Task 4: Claim Occurrence Prediction\n\n")
        f.write("## Binary Classification — Will This Policy Result in a Claim?\n\n")
        f.write("### June 17, 2025 | Firaol Driba\n\n")

        f.write("## 📊 Executive Summary\n")
        f.write("Built and evaluated two classification models to predict whether a policy will result in at least one claim.\n\n")

        f.write("## 🧪 Key Findings\n")
        for res in results:
            f.write(f"### {res['model_name']}\n")
            f.write(f"- Accuracy: {res['accuracy']:.4f}\n")
            f.write(f"- ROC AUC: {res['roc_auc']:.4f}\n") # Include ROC AUC in report
            f.write("```\n")
            f.write(res['report'])
            f.write("\n```\n\n")

        f.write("\n## 🧠 Top Features Influencing Claim Probability\n")
        f.write("- Vehicle age\n")
        f.write("- Province (e.g., Gauteng)\n")
        f.write("- Make (e.g., Toyota vs Mercedes-Benz)\n")
        f.write("- Cubic capacity\n")
        f.write("- Kilowatts\n")

        f.write("\n## 📌 Business Recommendations\n")
        f.write("- Use classification model to identify high-risk profiles before underwriting\n")
        f.write("- Adjust premiums for vehicles with higher predicted risk\n")
        f.write("- Recommend tracking devices or alarms for high-risk customers\n")
        f.write("- Combine with severity model to define overall risk profile\n")

    print(f"[OK] Report saved to {REPORT_FILE}")

# -----------------------------
# 📁 Main Execution
# -----------------------------
def main():
    # Load and clean data
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # Debug: Show class balance
    print("\n🔢 Total class distribution:")
    print(df['HadClaim'].value_counts())

    # Prepare features
    try:
        X_train, X_test, y_train, y_test, feature_names = prepare_features(df)

        # Balance training set
        X_resampled, y_resampled = resample_data(X_train, y_train)

    except ValueError as e:
        print(f"[ERROR] {str(e)}")
        print("💡 Try balancing your data before proceeding.")
        return

    # Define models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LogisticRegression": LogisticRegression(random_state=42, solver='liblinear') # Added Logistic Regression
    }

    results = []
    for name, model in models.items():
        result = train_and_evaluate(model, name, X_resampled, X_test, y_resampled, y_test)
        results.append(result)

    # Interpret best-performing model (if tree-based)
    # Using ROC AUC to select best classification model for interpretation
    best_model = max(results, key=lambda x: x['roc_auc'])
    print(f"\n🧠 Interpreting best model (based on ROC AUC): {best_model['model_name']}")

    # Only perform SHAP interpretation for tree-based classification models
    if isinstance(best_model['model'], (RandomForestClassifier, XGBClassifier)):
        interpret_model(best_model, X_test, feature_names)
    else:
        print(f"\nSkipping SHAP interpretation for {best_model['model_name']} as it's not a tree-based model supported by TreeExplainer.")


    # Generate final report
    generate_report(results)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    main()
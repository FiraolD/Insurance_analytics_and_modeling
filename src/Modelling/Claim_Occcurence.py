# File: src/modeling/claim_occurrence_classifier.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression # Import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_score, recall_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from insurance_analytics.targets import add_claim_targets

# -----------------------------
# 🔧 Configuration
# -----------------------------
# Update DATA_PATH to the correct absolute path
DATA_PATH = os.getenv("INSURANCE_DATA_PATH", "Data/MachineLearningRating_v3.txt")
MISSING_CLAIMS_POLICY = os.getenv("INSURANCE_MISSING_CLAIMS_POLICY", "unknown")
SEP = '|'

PLOT_DIR = "Reports/task4/plots"
REPORT_FILE = "Reports/task4/claim_occurrence_report.md"

os.makedirs(PLOT_DIR, exist_ok=True)

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

    # Keep the source outcome intact and classify missingness explicitly.
    # The source data contract must be verified before missing means no claim.
    df = add_claim_targets(df, missing_claims=MISSING_CLAIMS_POLICY)

    # 🔍 Print some sample rows to debug
    print("\n🔍 Sample HadClaim values:")
    print(df[['TotalClaims', 'HadClaim']].head(10))
    # Fill categorical columns safely
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    # Do not impute numeric columns here: target labels must remain untouched.

    print("[OK] Data cleaned and prepared.")
    return df

# -----------------------------
# 🧱 Build Train/Test Splits with Balancing
# -----------------------------
def prepare_features(df):
    """Split raw observations first, then fit preprocessing on training rows."""
    print("\n[INFO] Preparing features for modeling...")

    base_features = [
        'make', 'Model', 'Province', 'PostalCode',
        'Gender', 'MaritalStatus', 'VehicleType',
        'RegistrationYear', 'kilowatts', 'cubiccapacity',
        'SumInsured', 'CoverType'
    ]
    target = 'HadClaim'
    eligible = df.loc[df[target].notna()].copy()
    missing_features = [column for column in base_features if column not in eligible.columns]
    if missing_features:
        raise KeyError(f"Required model features are missing: {missing_features}")
    X = eligible[base_features]
    y = eligible[target].astype(int)

    print("\n[INFO] Verified claim-label distribution:")
    print(y.value_counts(dropna=False))
    if y.nunique() < 2:
        raise ValueError(
            "Only one verified claim class is available. Confirm the source-data "
            "meaning of missing TotalClaims before changing the missing-label policy."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler(with_mean=False)),
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=10)),
        ]), categorical_features),
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    cat_pipeline = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_pipeline.named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(encoded_cat_names)

    print(f"[OK] Feature shapes after split: {X_train_processed.shape}, {X_test_processed.shape}")
    return X_train_processed, X_test_processed, y_train, y_test, feature_names
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
    report = classification_report(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    average_precision = average_precision_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print(f"✅ {name} trained.")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average precision (PR-AUC summary): {average_precision:.4f}")
    print(f"Precision at threshold 0.5: {precision:.4f}; recall: {recall:.4f}")
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
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "precision": precision,
        "recall": recall
    }

# -----------------------------
# 🧠 Interpret Best Model Using SHAP
# -----------------------------
def interpret_model(model_info, X_test, feature_names):
    """Interpret model using SHAP values."""
    print("\n🧠 Interpreting model with SHAP...")

    # Explain a bounded sample only; densifying a full one-hot test matrix can
    # exhaust memory on this high-cardinality insurance dataset.
    X_sample = X_test[:500]
    try:
        X_test_dense = X_sample.toarray()
    except AttributeError:
        X_test_dense = X_sample

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
        f.write("Models are evaluated on an untouched holdout set. Precision and recall are shown at threshold 0.5; threshold selection and probability calibration require separate validation.\n\n")

        f.write("## 🧪 Key Findings\n")
        for res in results:
            f.write(f"### {res['model_name']}\n")
            f.write(f"- Accuracy: {res['accuracy']:.4f}\n")
            f.write(f"- ROC AUC: {res['roc_auc']:.4f}\n")
            f.write(f"- Average precision: {res['average_precision']:.4f}\n")
            f.write(f"- Precision at threshold 0.5: {res['precision']:.4f}\n")
            f.write(f"- Recall at threshold 0.5: {res['recall']:.4f}\n")
            f.write("```\n")
            f.write(res['report'])
            f.write("\n```\n\n")

        f.write("\n## 🧠 Top Features Influencing Claim Probability\n")
        f.write("- Review the generated SHAP plots for model-specific feature attributions; importance does not imply causality.\n")

        f.write("\n## 📌 Business Recommendations\n")
        f.write("- Treat outputs as exploratory until target semantics, threshold selection, calibration, fairness, and out-of-time performance are validated.\n")
        f.write("- Do not make underwriting or pricing decisions from this model without independent governance and actuarial review.\n")

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
# File: src/modelling/full_model_pipeline.py

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

import shap
import matplotlib.pyplot as plt
import seaborn as sns
from insurance_analytics.targets import add_claim_targets

# -----------------------------
# 🔧 Configuration
# -----------------------------
DATA_PATH = os.getenv("INSURANCE_DATA_PATH", "Data/MachineLearningRating_v3.txt")
MISSING_CLAIMS_POLICY = os.getenv("INSURANCE_MISSING_CLAIMS_POLICY", "unknown")
SEP = '|'

PLOT_DIR = "Reports/task4/plots"
REPORT_FILE = "Reports/task4/final_model_report.md"

os.makedirs(PLOT_DIR, exist_ok=True)

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

    # Normalize financial source columns without filling missing outcomes.
    for column in ('TotalPremium', 'TotalClaims'):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    # Construct claim outcomes before any feature cleaning or imputation.
    df = add_claim_targets(df, missing_claims=MISSING_CLAIMS_POLICY)

    # Preserve signed financial values for review; do not silently apply abs().
    # Fill categorical columns only; learned numeric imputation is fitted
    # on training data only in each task-specific pipeline.
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    # Outcome-derived ratios are omitted from the feature frame to reduce the
    # risk of accidental target leakage in later modelling changes.

    print("[OK] Data cleaned and prepared.")
    return df

# -----------------------------
# 🧱 Build Train/Test Splits
# -----------------------------
def prepare_features(df):
    """Build task-specific splits and fit each preprocessor on training data only."""
    print("\n[INFO] Preparing task-specific datasets...")

    base_features = [
        'make', 'Model', 'Province', 'PostalCode',
        'Gender', 'MaritalStatus', 'VehicleType',
        'RegistrationYear', 'kilowatts', 'cubiccapacity',
        'SumInsured', 'CoverType'
    ]
    missing_features = [column for column in base_features if column not in df.columns]
    if missing_features:
        raise KeyError(f"Required model features are missing: {missing_features}")

    def split_and_transform(X, y, task_name, stratify=False):
        if len(y) < 2:
            raise ValueError(f"Not enough eligible observations for {task_name}.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if stratify else None
        )
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=10)),
            ]), categorical_features),
        ])
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        encoded_cat_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names = numeric_features + list(encoded_cat_names)
        print(f"[OK] {task_name}: train={X_train_processed.shape}, test={X_test_processed.shape}")
        return X_train_processed, X_test_processed, y_train, y_test, feature_names

    severity_mask = df['ClaimSeverityTarget'].notna()
    severity_data = df.loc[severity_mask]
    severity = split_and_transform(
        severity_data[base_features], severity_data['ClaimSeverityTarget'], 'severity'
    )

    premium_target = pd.to_numeric(df['TotalPremium'], errors='coerce')
    premium_mask = premium_target.notna() & premium_target.ge(0)
    premium_data = df.loc[premium_mask]
    premium = split_and_transform(
        premium_data[base_features], premium_target.loc[premium_mask], 'historical premium'
    )

    claim_mask = df['HadClaim'].notna()
    claim_data = df.loc[claim_mask]
    claim_y = claim_data['HadClaim'].astype(int)
    if claim_y.nunique() < 2:
        print(
            "[WARN] Skipping claim-occurrence model: fewer than two verified classes. "
            "Verify the source data dictionary before interpreting missing claim "
            "amounts as no-claim outcomes."
        )
        claim = None
    else:
        claim = split_and_transform(
            claim_data[base_features], claim_y, 'claim occurrence', stratify=True
        )

    return {'severity': severity, 'premium': premium, 'claim': claim}

# -----------------------------
# 🚀 Train and Evaluate Models
# -----------------------------
def train_severity_model(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate models for predicting claim severity."""
    print("\n[🚀] Training Claim Severity Models...")
    
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"\n🧪 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"✅ {name} trained.")
        print(f"RMSE: {rmse:.2f}, R² Score: {r2:.4f}")
        results.append({
            'model_name': f'Severity_{name}',
            'model': model,
            'rmse': rmse,
            'r2': r2
        })

        plot_predictions(y_test, y_pred, name, "claim_severity", PLOT_DIR)

    best_model = max(results, key=lambda x: x['r2'])
    interpret_model(best_model, X_test, feature_names, "claim_severity")

    return results

# -----------------------------
# 💰 Train Premium Prediction Model
# -----------------------------
def train_premium_model(X_train, X_test, y_train, y_test, feature_names):
    """Train models for predicting premium amount."""
    print("\n[🚀] Training Premium Prediction Models...")

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"\n🧪 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"✅ {name} trained.")
        print(f"RMSE: {rmse:.2f}, R² Score: {r2:.4f}")
        results.append({
            'model_name': f'Premium_{name}',
            'model': model,
            'rmse': rmse,
            'r2': r2
        })

        plot_predictions(y_test, y_pred, name, "premium_prediction", PLOT_DIR)

    best_model = max(results, key=lambda x: x['r2'])
    interpret_with_shap(best_model['model'], X_test, feature_names, "premium")

    return results

# -----------------------------
# 🧮 Train Claim Occurrence Classifier
# -----------------------------
def train_claim_classifier(X_train, X_test, y_train, y_test, feature_names):
    """Train classification model to predict if a claim occurred."""
    print("\n[🚀] Training Claim Occurrence Classifier...")

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = []

    for name, model in models.items():
        print(f"\n🧪 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"✅ {name} trained.")
        print(f"Accuracy: {acc:.4f}")
        print(report)

        results.append({
            'model_name': f'Claim_Occurrence_{name}',
            'model': model,
            'accuracy': acc,
            'report': report
        })

    best_model = max(results, key=lambda x: x['accuracy'])
    interpret_with_shap(best_model['model'], X_test, feature_names, "claim_occurrence")

    return results

# -----------------------------
# 📈 Plot Predictions
# -----------------------------
def plot_predictions(y_true, y_pred, model_name, prefix, plot_dir):
    """Plot predictions vs actuals."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f"{model_name} Predictions vs Actuals")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{prefix}_{model_name.lower().replace(' ', '_')}_predictions.png"))
    plt.close()

# -----------------------------
# 🧠 Interpret Best Model Using SHAP
# -----------------------------
def interpret_model(model_info, X_test, feature_names, context):
    """Interpret model using SHAP values."""
    print(f"\n🧠 Interpreting model ({context})...")

    # Check if X_test is sparse and convert to dense
    # Keep interpretation bounded to avoid densifying a huge one-hot matrix.
    X_sample = X_test[:500]
    try:
        X_test_dense = X_sample.toarray()
    except AttributeError:
        X_test_dense = X_sample

    explainer = shap.Explainer(model_info["model"], X_test_dense)
    shap_values = explainer(X_test_dense)

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_dense, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(PLOT_DIR, f"shap_summary_{context}.png"), bbox_inches='tight')
    plt.close()

    # Bar plot of feature importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_dense, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig(os.path.join(PLOT_DIR, f"shap_bar_{context}.png"), bbox_inches='tight')
    plt.close()

    print(f"[OK] SHAP analysis for {context} completed.")

# -----------------------------
# 📝 Generate Final Report
# -----------------------------
def generate_final_report(sev_results, pre_results, cla_results):
    """Generate markdown report summarizing findings."""
    print("\n📄 Generating final model comparison report...")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("# 🤖 Task 4: Predictive Modeling\n\n")
        f.write("## Car Insurance Risk Analytics — June 17, 2025\n\n")
        f.write("## 📊 Executive Summary\n")
        f.write("Built and evaluated multiple machine learning models to support dynamic pricing and segmentation.\n\n")

        f.write("## 🧪 Key Findings\n")
        f.write("- **Claim Severity Prediction**\n")
        for res in sev_results:
            f.write(f"  - {res['model_name']}: RMSE={res['rmse']:.2f}, R²={res['r2']:.4f}\n")
        f.write("- **Premium Prediction**\n")
        for res in pre_results:
            f.write(f"  - {res['model_name']}: RMSE={res['rmse']:.2f}, R²={res['r2']:.4f}\n")
        f.write("- **Claim Occurrence Classification**\n")
        if not cla_results:
            f.write("  - Skipped: insufficient verified outcome classes under the configured missing-value policy.\n")
        for res in cla_results:
            f.write(f"  - {res['model_name']}: Accuracy={res['accuracy']:.4f}\n")
            f.write(res['report'] + "\n")

        f.write("\n## 🧠 Top Features Influencing Predictions\n")
        f.write("From SHAP analysis:\n")
        f.write("  - See generated SHAP plots for task-specific attributions; these are not causal effects.\n")

        f.write("\n## 📌 Business Recommendations\n")
        f.write("- Treat model comparisons as exploratory until reproduced with verified targets and appropriate holdout metrics.\n")
        f.write("- Historical premium prediction is not an actuarial indication of adequate premium.\n")
        f.write("- Any expected-cost estimate must align claim frequency, conditional severity, exposure, currency, and observation unit.\n")

    print(f"[OK] Report saved to {REPORT_FILE}")

# -----------------------------
# 📁 Main Execution
# -----------------------------
def main():
    # Load and clean data
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # Prepare features for each task
    feature_sets = prepare_features(df)

    # Train each target with its own leakage-safe preprocessing pipeline.
    sev_results = train_severity_model(*feature_sets['severity'])
    pre_results = train_premium_model(*feature_sets['premium'])
    cla_results = (
        train_claim_classifier(*feature_sets['claim'])
        if feature_sets['claim'] is not None
        else []
    )

    # Generate final report
    generate_final_report(sev_results, pre_results, cla_results)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()
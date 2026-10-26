# File: eda_cleaner.py
# Combined EDA + Cleaning Script

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from insurance_analytics.targets import add_claim_targets, summarize_claim_amounts

DATA_PATH = os.getenv("INSURANCE_DATA_PATH", "Data/MachineLearningRating_v3.txt")
MISSING_CLAIMS_POLICY = os.getenv("INSURANCE_MISSING_CLAIMS_POLICY", "unknown")

# --------------------------
# 🔧 DATA LOADING & CLEANING
# --------------------------

def load_data(filepath):
    """Load dataset with low_memory=False."""
    df = pd.read_csv(filepath, sep='|', low_memory=False)
    print("[OK] Data loaded successfully.")
    return df

def handle_missing_values(df):
    """Report missingness without silently imputing outcomes or features."""
    missing = df.isna().sum().sort_values(ascending=False)
    print("[INFO] Columns with the most missing values:")
    print(missing[missing.gt(0)].head(20).to_string() or "No missing values found.")
    return df

def fix_negative_values(df):
    """Report negative financial values without changing their accounting sign."""
    for column in ("TotalPremium", "TotalClaims"):
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            count = int(values.lt(0).sum())
            print(f"[DATA QUALITY] {column}: {count} negative values retained for review.")
    return df

def convert_to_categorical(df):
    """Convert selected columns to category type."""
    print("[INFO] Converting columns to categorical...")

    cols_to_category = ['Gender', 'MaritalStatus', 'Province', 'PostalCode',
                        'CoverType', 'VehicleType', 'make', 'Model']
    for col in cols_to_category:
        if col in df.columns:
            df[col] = df[col].astype('category')

    print("[OK] Columns converted to categorical.")
    return df

def calculate_loss_ratio(df):
    """Calculate loss ratio only where premium and claim amount are defined."""
    premium = pd.to_numeric(df["TotalPremium"], errors="coerce")
    claims = df["ClaimAmountObserved"].copy()
    if MISSING_CLAIMS_POLICY == "no_claim":
        claims = claims.fillna(0)
    df["LossRatio"] = (claims / premium).where(premium.gt(0))
    return df

def clean_and_prepare(df):
    """Normalize source fields and preserve unknown outcomes for analysis."""
    df = df.copy()
    raw_claim_quality = summarize_claim_amounts(df, "TotalClaims")
    for column in ("TotalPremium", "TotalClaims"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = add_claim_targets(df, missing_claims=MISSING_CLAIMS_POLICY)
    df = handle_missing_values(df)
    df = fix_negative_values(df)
    df = convert_to_categorical(df)

    if "TransactionMonth" in df.columns:
        df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce")

    df = calculate_loss_ratio(df)
    print("Claim amount data-quality summary:")
    for key, value in raw_claim_quality.items():
        print(f"  {key}: {value}")
    return df

# -------------------
# 📊 VISUALIZATIONS
# -------------------

def plot_loss_ratio_by_province(df):
    plt.figure(figsize=(16, 5))
    sns.barplot(data=df, x="Province", y="LossRatio", estimator='mean', errorbar=None)
    plt.title("Average Loss Ratio by Province")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_loss_ratio_by_vehicle_type(df):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="VehicleType", y="LossRatio", estimator='mean', errorbar=None)
    plt.title("Average Loss Ratio by Vehicle Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_loss_ratio_by_gender(df):
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="Gender", y="LossRatio", estimator='mean', errorbar=None)
    plt.title("Average Loss Ratio by Gender")
    plt.tight_layout()
    plt.show()

def plot_claim_severity_by_make(df):
    claim_rows = df.loc[df["ClaimSeverityTarget"].notna()]
    claim_severity = claim_rows.groupby("make")["ClaimSeverityTarget"].mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 5))
    sns.barplot(x=claim_severity.index, y=claim_severity.values)
    plt.title("Average Claim Severity by Vehicle Make (Top 15)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_premium_vs_claims(df):
    plt.figure(figsize=(8, 6))
    plot_data = df.loc[df["TotalPremium"].gt(0) & df["ClaimAmountObserved"].gt(0)]
    sns.scatterplot(data=plot_data, x="TotalPremium", y="ClaimAmountObserved", hue="VehicleType", alpha=0.7)
    plt.title("Premium vs Positive Observed Claims")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

def plot_financial_distributions(df):
    financial_vars = ["TotalPremium", "TotalClaims", "LossRatio", "CustomValueEstimate"]

    plt.figure(figsize=(12, 8))
    for i, var in enumerate(financial_vars):
        plt.subplot(2, 2, i + 1)
        sns.histplot(df[var], kde=True)
        plt.title(f"Distribution of {var}")
    plt.tight_layout()
    plt.show()

def plot_outliers(df):
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["TotalClaims", "CustomValueEstimate"]])
    plt.title("Outliers in TotalClaims and CustomValueEstimate")
    plt.tight_layout()
    plt.show()

def plot_monthly_trend(df):
    if 'TransactionMonth' not in df.columns:
        print("[ERROR] TransactionMonth column missing for monthly trend.")
        return

    if MISSING_CLAIMS_POLICY == "unknown":
        print("[WARN] Monthly claim totals are skipped because missing claim amounts remain unknown.")
        return

    df['month'] = df['TransactionMonth'].dt.to_period("M").astype(str)

    monthly = df.groupby("month").agg({
        "ClaimAmountObserved": "sum",
        "LossRatio": "mean",
        "PolicyID": "nunique"
    }).rename(columns={"PolicyID": "UniquePolicies"})

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly, x=monthly.index, y="ClaimAmountObserved", label="Claim Amounts")
    sns.lineplot(data=monthly, x=monthly.index, y="LossRatio", label="Avg Loss Ratio")
    plt.xticks(rotation=45)
    plt.title("Monthly Claim Frequency and Severity")
    plt.tight_layout()
    plt.show()

def plot_top_bottom_makes(df):
    claim_rows = df.loc[df["ClaimSeverityTarget"].notna()]
    top_makes = claim_rows.groupby("make")["ClaimSeverityTarget"].sum().sort_values(ascending=False)

    plt.figure(figsize=(12, 5))
    top_makes.head(10).plot(kind='bar', color='tomato')
    plt.title("Top 10 Vehicle Makes with Highest Total Claims")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    top_makes.tail(10).plot(kind='bar', color='seagreen')
    plt.title("Bottom 10 Vehicle Makes with Lowest Total Claims")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------------
# 🏁 MAIN FUNCTION
# ---------------

def main():
    # Define path
    file_path = DATA_PATH

    # Load and clean data
    df = load_data(file_path)
    df = clean_and_prepare(df)

    # Run visualizations
    plot_loss_ratio_by_province(df)
    plot_loss_ratio_by_vehicle_type(df)
    plot_loss_ratio_by_gender(df)
    plot_claim_severity_by_make(df)
    plot_premium_vs_claims(df)
    plot_financial_distributions(df)
    plot_outliers(df)
    plot_monthly_trend(df)
    plot_top_bottom_makes(df)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main()
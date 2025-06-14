# File: eda_cleaner.py
# Combined EDA + Cleaning Script

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --------------------------
# 🔧 DATA LOADING & CLEANING
# --------------------------

def load_data(filepath):
    """Load dataset with low_memory=False."""
    df = pd.read_csv(filepath, sep='|', low_memory=False)
    print("[OK] Data loaded successfully.")
    return df

def handle_missing_values(df):
    """Fill or drop missing values."""
    print("[INFO] Handling missing values...")

    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    print("[OK] Missing values handled.")
    return df

def fix_negative_values(df):
    """Fix negative TotalPremium and TotalClaims."""
    print("[INFO] Fixing negative premium and claim values...")

    if 'TotalPremium' in df.columns:
        df['TotalPremium'] = df['TotalPremium'].abs()
    if 'TotalClaims' in df.columns:
        df['TotalClaims'] = df['TotalClaims'].abs()

    print("[OK] Negative values fixed.")
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
    """Calculate Loss Ratio = Claims / Premium."""
    print("[INFO] Calculating loss ratio...")

    if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
        df['LossRatio'] = np.where(
            df['TotalPremium'] > 0,
            df['TotalClaims'] / df['TotalPremium'],
            np.nan
        )
        print("[OK] Loss ratio calculated.")
    else:
        print("[ERROR] Required columns missing for loss ratio calculation.")

    return df

def clean_and_prepare(df):
    """Run all cleaning steps in sequence."""
    df = handle_missing_values(df)
    df = fix_negative_values(df)
    df = convert_to_categorical(df)
    df = calculate_loss_ratio(df)

    # Convert TransactionMonth to datetime
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

    # Drop rows with critical missing values
    essential_cols = ["TransactionMonth", "Province", "VehicleType", "Gender",
                    "make", "Model", "TotalPremium", "TotalClaims",
                    "LossRatio", "CustomValueEstimate"]
    df = df.dropna(subset=essential_cols)
    print("[OK] Critical missing rows dropped for visualization.")

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
    claim_severity = df.groupby("make")["TotalClaims"].mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 5))
    sns.barplot(x=claim_severity.index, y=claim_severity.values)
    plt.title("Average Claim Severity by Vehicle Make (Top 15)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_premium_vs_claims(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="TotalPremium", y="TotalClaims", hue="VehicleType", alpha=0.7)
    plt.title("Premium vs Claims")
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

    df['month'] = df['TransactionMonth'].dt.to_period("M").astype(str)

    monthly = df.groupby("month").agg({
        "TotalClaims": "sum",
        "LossRatio": "mean",
        "PolicyID": "nunique"
    }).rename(columns={"PolicyID": "UniquePolicies"})

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly, x=monthly.index, y="TotalClaims", label="Total Claims")
    sns.lineplot(data=monthly, x=monthly.index, y="LossRatio", label="Avg Loss Ratio")
    plt.xticks(rotation=45)
    plt.title("Monthly Claim Frequency and Severity")
    plt.tight_layout()
    plt.show()

def plot_top_bottom_makes(df):
    top_makes = df.groupby("make")["TotalClaims"].sum().sort_values(ascending=False)

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

# -------------------
# 🏁 MAIN FUNCTION
# -------------------

def main():
    # Define path
    file_path = 'D:/PYTHON PROJECTS/KIAM PROJECTS/Insurance_analytics_and_modeling/Data/MachineLearningRating_v3.txt'

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
# File: src/hypothesis_testing/hypothesis_tester.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
import os

# -----------------------------
# 🔧 Configuration
# -----------------------------
DATA_PATH = "Data/MachineLearningRating_v3.txt"
SEP = '|'

PLOT_DIR = "Reports/task3-visualizations"
OUTPUT_FILE = "Reports/output/task3_report.md"

# -----------------------------
# 📥 Load Dataset
# -----------------------------
def load_data(path):
    """Load dataset using low_memory=False."""
    df = pd.read_csv(path, sep=SEP, low_memory=False)
    print("[OK] Data loaded successfully.")
    return df

# -----------------------------
# 🧹 Clean & Prepare Data
# -----------------------------
def clean_data(df):
    """Clean data by handling missing values and calculating metrics."""
    print("[INFO] Cleaning data...")

    # Fix negative values
    if 'TotalPremium' in df.columns:
        df['TotalPremium'] = df['TotalPremium'].abs()
    if 'TotalClaims' in df.columns:
        df['TotalClaims'] = df['TotalClaims'].abs()
        
    # convert 'Transaction Month ' to datetime
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce', format='mixed')

    # Fill categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    # Fill numerical columns
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Calculate Loss Ratio
    df['LossRatio'] = np.where(
        df['TotalPremium'] > 0,
        df['TotalClaims'] / df['TotalPremium'],
        np.nan
    )

    # Binary claim indicator
    df['HadClaim'] = df['TotalClaims'].notnull().astype(int)

    print("[OK] Data cleaned and prepared.")
    return df

# -----------------------------
# 📈 Generate and Save Plots
# -----------------------------
def save_visualizations(df):
    """Generate and save visualizations to Reports/task3-visualizations"""
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Loss ratio by province
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Province', y='LossRatio', data=df, estimator=np.mean, errorbar=None)
    plt.xticks(rotation=45)
    plt.title("Average Loss Ratio by Province")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "loss_ratio_by_province.png"))
    plt.close()

    # Claim severity by vehicle make
    top_makes = df.groupby('make')['TotalClaims'].mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 5))
    top_makes.plot(kind='bar', color='tomato')
    plt.title("Top 15 Vehicle Makes with Highest Claim Severity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "claim_severity_by_make.png"))
    plt.close()

    # Loss ratio by gender
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="Gender", y="LossRatio", estimator=np.mean, errorbar=None)
    plt.title("Average Loss Ratio by Gender")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "loss_ratio_by_gender.png"))
    plt.close()

    # Monthly trend (only if TransactionMonth is valid)
    if 'TransactionMonth' in df.columns and pd.api.types.is_datetime64_any_dtype(df['TransactionMonth']):
        df['month'] = df['TransactionMonth'].dt.to_period("M").astype(str)
        monthly = df.groupby("month")[['TotalClaims', 'LossRatio']].agg(['sum', 'mean'])
        monthly.columns = ['_'.join(col) for col in monthly.columns]

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=monthly, x=monthly.index, y="TotalClaims_sum", label="Total Claims")
        sns.lineplot(data=monthly, x=monthly.index, y="LossRatio_mean", label="Avg Loss Ratio")
        plt.xticks(rotation=45)
        plt.title("Monthly Claim Frequency and Severity")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "monthly_claim_trend.png"))
        plt.close()
    else:
        print("⚠️ TransactionMonth not available or not datetime — skipping monthly claim trend plot.")

# -----------------------------
# 📊 Statistical Hypotheses
# -----------------------------
def test_province_hypothesis(df):
    """Test if there's a significant difference in loss ratio between provinces."""
    print("\n🧪 Hypothesis: Risk differs across provinces")

    gauteng = df[df['Province'] == 'Gauteng']['LossRatio'].dropna()
    western_cape = df[df['Province'] == 'Western Cape']['LossRatio'].dropna()

    if len(gauteng) < 2 or len(western_cape) < 2:
        return {"name": "Risk differs across provinces", "result": False, "message": "⚠️ Not enough samples for province test"}

    t_stat, p_value = ttest_ind(gauteng, western_cape, equal_var=False)

    result = p_value < 0.05
    message = f"T-statistic: {t_stat:.4f}, P-value: {p_value:.6f}\n{'✅ Reject H₀: Significant difference in loss ratio between provinces.' if result else '❌ Fail to reject H₀: No significant difference in loss ratio.'}"

    return {"name": "Risk differs across provinces", "result": result, "message": message}

def test_zipcode_hypothesis(df):
    """Test if there's a significant difference in loss ratio between zipcodes."""
    print("\n🧪 Hypothesis: Risk differs between zipcodes")

    top_zipcodes = df['PostalCode'].value_counts().index[:2]
    zip_a = df[df['PostalCode'] == top_zipcodes[0]]['LossRatio'].dropna()
    zip_b = df[df['PostalCode'] == top_zipcodes[1]]['LossRatio'].dropna()

    if len(zip_a) < 2 or len(zip_b) < 2:
        return {"name": "Risk differs between zipcodes", "result": False, "message": "⚠️ Not enough samples for zipcode test"}

    f_stat, p_value = f_oneway(zip_a, zip_b)

    result = p_value < 0.05
    message = f"F-statistic: {f_stat:.4f}, P-value: {p_value:.6f}\n{'✅ Reject H₀: Significant difference in loss ratio between zipcodes.' if result else '❌ Fail to reject H₀: No significant difference in loss ratio.'}"

    return {"name": "Risk differs between zipcodes", "result": result, "message": message}

def test_gender_hypothesis(df):
    """Test if there's a significant risk difference between genders."""
    print("\n🧪 Hypothesis: Risk differs between genders")

    contingency_table = pd.crosstab(df['Gender'], df['HadClaim'])
    chi2, p, _, _ = chi2_contingency(contingency_table)

    result = p < 0.05
    message = f"Chi-squared: {chi2:.4f}, P-value: {p:.6f}\n{'✅ Reject H₀: Significant risk difference between genders.' if result else '❌ Fail to reject H₀: No significant risk difference.'}"

    return {"name": "Risk differs between genders", "result": result, "message": message}

def test_profit_zipcode_hypothesis(df):
    """Test if there's a significant profit margin difference between zipcodes."""
    print("\n🧪 Hypothesis: Profit margin differs between zipcodes")

    df['ProfitMargin'] = df['TotalPremium'] - df['TotalClaims']

    top_zipcodes = df['PostalCode'].value_counts().index[:2]
    zip_a = df[df['PostalCode'] == top_zipcodes[0]]['ProfitMargin'].dropna()
    zip_b = df[df['PostalCode'] == top_zipcodes[1]]['ProfitMargin'].dropna()

    if len(zip_a) < 2 or len(zip_b) < 2:
        return {"name": "Profit margin differs between zipcodes", "result": False, "message": "⚠️ Not enough samples for profit margin test"}

    t_stat, p_value = ttest_ind(zip_a, zip_b, equal_var=False)

    result = p_value < 0.05
    message = f"T-statistic: {t_stat:.4f}, P-value: {p_value:.6f}\n{'✅ Reject H₀: Significant profit margin difference between zipcodes.' if result else '❌ Fail to reject H₀: No significant profit margin difference.'}"

    return {"name": "Profit margin differs between zipcodes", "result": result, "message": message}

# -----------------------------
# 📝 Generate Markdown Report
# -----------------------------
def generate_report(df):
    """Run all tests and write results to markdown report"""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("\n📄 Running hypothesis tests...")
    results = [
        test_province_hypothesis(df),
        test_zipcode_hypothesis(df),
        test_gender_hypothesis(df),
        test_profit_zipcode_hypothesis(df)
    ]

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# 🧪 Task 3: A/B Hypothesis Testing\n\n")
        f.write("Statistical analysis of car insurance risk drivers:\n\n")

        for res in results:
            f.write(f"## {res['name']}\n")
            f.write(res['message'] + "\n\n")

        f.write("## ✅ Business Recommendations\n")
        any_rejected = False
        for res in results:
            if res['result']:
                any_rejected = True
                f.write("- " + res['message'].split('\n')[0] + "\n")

        if not any_rejected:
            f.write("- No significant risk drivers found — consider other variables.\n")

    print(f"[OK] Report saved to {OUTPUT_FILE}")

# -----------------------------
#  Main Execution
# -----------------------------
def main():
    # Load and clean data
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # Generate plots
    save_visualizations(df)

    # Run tests and generate report
    generate_report(df)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")  # Suppress harmless warnings
    main()
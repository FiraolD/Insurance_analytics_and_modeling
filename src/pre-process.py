# File: src/pre-processor.py
#this file is used to display basic information for the data

import pandas as pd
import os

def load_data(filepath):
    """Load dataset with low_memory=False to handle mixed dtypes."""
    df = pd.read_csv(filepath, sep='|', low_memory=False)
    print("[OK] Data loaded successfully.")
    return df

def basic_info(df):
    """Print basic info about the DataFrame."""
    print("\n🔢 Shape of the data:", df.shape)
    print("\n📊 First 5 rows:")
    print(df.head())
    print("\n🧾 Data types and non-null counts:")
    print(df.info())
    print("\n🧮 Summary statistics:")
    print(df.describe(include='all'))

def missing_values(df):
    """Display missing value percentages per column."""
    print("\n🚫 Missing Values per Column:")
    missing = df.isnull().sum()
    percent_missing = (missing / len(df)) * 100
    missing_table = pd.DataFrame({'Missing Count': missing,
                                'Missing %': percent_missing})
    print(missing_table.sort_values(by='Missing %', ascending=False).head(20))

def top_categorical_values(df):
    """Show top values in categorical columns."""
    print("\n📍 Unique & Top Values in Categorical Columns:")
    for col in df.select_dtypes(include='object').columns:
        print(f"\nUnique values in '{col}':")
        print(df[col].value_counts(dropna=False).head(10))

if __name__ == '__main__':
    # Define paths
    #project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join('D:/PYTHON PROJECTS/KIAM PROJECTS/Insurance_analytics_and_modeling', 'Data', 'MachineLearningRating_v3.txt')
    #df = pd.read_csv("C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/Insurance-Risk-Analytics/Data/sentiment_reviews2.csv", )  # or whatever the filename is
    # Load data
    df = load_data(data_path)

    # Run EDA functions
    basic_info(df)
    missing_values(df)
    top_categorical_values(df)
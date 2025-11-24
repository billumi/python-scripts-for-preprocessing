# eda.py
import pandas as pd

def dataset_summary(df: pd.DataFrame):

    print("\n===== DATAFRAME OVERVIEW =====")
    print(df.info())
    print(df.head())

    print("\n===== SHAPE =====")
    print(df.shape)
    
    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())
    
    print("\n===== DATA TYPES =====")
    print(df.dtypes)
    
    print("\n===== DESCRIPTIVE STATS =====")
    print(df.describe(include="all"))

def correlation_matrix(df: pd.DataFrame):
    num_df = df.select_dtypes(include=["int64", "float64"])
    return num_df.corr()


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all")

def fill_missing_numeric(df: pd.DataFrame, strategy="median"):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    imputer = SimpleImputer(strategy=strategy)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def fill_missing_categorical(df: pd.DataFrame, strategy="most_frequent"):
    cat_cols = df.select_dtypes(include=['object']).columns
    imputer = SimpleImputer(strategy=strategy)
    df[cat_cols] = imputer.fit_transform(df[cat_cols])
    return df

def cap_outliers_iqr(df: pd.DataFrame, cols=None):
    if cols is None:
        cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = np.where(df[col] < lower, lower,
                           np.where(df[col] > upper, upper, df[col]))
    return df

def normalize_text(df: pd.DataFrame): # Cleaning the categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return df


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def encode_categorical(df: pd.DataFrame, categorical_cols = None):
    if not categorical_cols:
        categorical_cols = df.select_dtypes(include="object").columns

    if len(categorical_cols) == 0: 
            return df
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def scale_numeric(df: pd.DataFrame):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def create_interaction_terms(df: pd.DataFrame, col_pairs: list):
    for col1, col2 in col_pairs:
        df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    return df

def create_ratio_features(df: pd.DataFrame, num_pairs: list):
    for numerator, denominator in num_pairs:
        df[f"{numerator}_to_{denominator}"] = (
            df[numerator] / df[denominator].replace(0, 1)
        )
    return df

def bin_numeric(df: pd.DataFrame, col: str, bins: int = 5):
    df[f"{col}_bin"] = pd.qcut(df[col], q=bins, duplicates="drop")
    return df

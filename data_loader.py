import pandas as pd
from typing import List


def load_csv(path: str, columns: List[str] = None) -> pd.DataFrame:
    '''
        Load data from the csv file for the given columns into a pandas dataframe
    '''
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="ISO-8859-1")
    except FileNotFoundError:
        raise Exception(f"File not found: {path}")

    if columns:
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise Exception(f"Missing columns: {missing_cols}")

        df = df[columns]

    return df


def load_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise Exception(f"Unable to load parquet file: {e}")

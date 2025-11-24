# main.py
from data_loader import load_csv
from data_cleaning import (
    remove_duplicates,
    drop_empty_rows,
    fill_missing_numeric,
    fill_missing_categorical,
    cap_outliers_iqr,
    normalize_text
)
from feature_engineering import encode_categorical, scale_numeric, create_interaction_terms 

from eda import dataset_summary
from plots import plot_histograms, plot_correlation_heatmap

def run_pipeline(path: str):
    print("\nLoading data...")
    df = load_csv(path)

    # EDA before cleaning
    print("\nInitial info:")
    dataset_summary(df)
    plot_histograms(df)

    df = remove_duplicates(df)
    df = drop_empty_rows(df)
    df = fill_missing_numeric(df)
    df = fill_missing_categorical(df)
    df = normalize_text(df)
    df = cap_outliers_iqr(df)

    # Preprocessing: encode + scale
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = encode_categorical(df, categorical_cols)
    df = scale_numeric(df)

    # Feature Engineering Example
    df = create_interaction_terms(df, [("feature1", "feature2")])

    # EDA after processing
    plot_correlation_heatmap(df)

    print("\nFinal cleaned dataset:")
    print_info(df)

    return df


if __name__ == "__main__":
    cleaned_df = run_pipeline("input.csv")
    print(cleaned_df.head())

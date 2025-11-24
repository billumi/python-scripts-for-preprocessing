# plots.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(include=["float64", "int64"]).columns
    
    df[cols].hist(figsize=(12, 10), bins=30)
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(include=["float64", "int64"]).columns
    
    plt.figure(figsize=(12, 8))
    df[cols].plot(kind='box')
    plt.xticks(rotation=45)
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=False, cmap="viridis")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_pairplot(df, cols=None):
    if cols:
        sns.pairplot(df[cols])
    else:
        sns.pairplot(df)
    plt.show()

def plot_categorical_counts(df, col):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=45)
    plt.show()

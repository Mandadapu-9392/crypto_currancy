"""
Statistics and Trends Assignment

Crypto-50 Market Analysis
"""

import os
import warnings
from corner import corner  # kept as required (template import)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

def plot_relational_plot(df):
    """Line plot of normalized closing prices (log scale)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    coins = ["BTC", "ETH", "BNB", "SOL"]

    for coin in coins:
        cdf = df[df["Symbol"] == coin].set_index("Date")["Close"]
        ax.plot(cdf.index, cdf / cdf.iloc[0], label=coin)

    ax.set_yscale("log")
    ax.set_title("Normalized Closing Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (log scale)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("relational_plot.png")
    plt.close()

def plot_categorical_plot(df):
    """Bar chart of top 20 annualised mean returns."""
    ann_return = (
        df.groupby("Symbol")["LogReturn"]
        .agg(mean=np.mean, count="count")
        .query("count >= 365")
        .assign(Annualised=lambda x: x["mean"] * 365)
        .sort_values("Annualised", ascending=False)
        .head(20)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(ann_return["Symbol"], ann_return["Annualised"])

    ax.set_title("Top 20 Annualised Mean Log-Returns")
    ax.set_xlabel("Coin")
    ax.set_ylabel("Annualised Log-Return")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("categorical_plot.png")
    plt.close()

def plot_statistical_plot(df):
    """Boxplot of daily log-returns for top 10 coins."""
    top10 = (
        df.groupby("Symbol")["LogReturn"]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    box_data = df[df["Symbol"].isin(top10)]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=box_data, x="Symbol", y="LogReturn", ax=ax)

    ax.set_title("Distribution of Daily Log-Returns (Top 10 Coins)")
    ax.set_xlabel("Coin")
    ax.set_ylabel("Log-Return")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("statistical_plot.png")
    plt.close()

def statistical_analysis(df, col: str):
    """Return mean, std, skewness and excess kurtosis."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col], nan_policy="omit")
    excess_kurtosis = ss.kurtosis(df[col], nan_policy="omit")

    return mean, stddev, skew, excess_kurtosis

def preprocessing(df):
    """Clean and prepare crypto dataset."""
    print("Initial shape:", df.shape)

    df = df.drop_duplicates()
    df = df[df["Close"] > 0]

    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    df["LogReturn"] = df.groupby("Symbol")["Close"].transform(
        lambda x: np.log(x / x.shift(1))
    )

    df = df.dropna(subset=["LogReturn"])

    print("After preprocessing:", df.shape)
    print("\nSummary statistics:")
    print(df.describe())
    print("\nCorrelation matrix:")
    print(df.corr(numeric_only=True))

    return df

def writing(moments, col):
    """Print statistical interpretation."""
    print(f"\nFor the attribute {col}:")
    print(
        f"Mean = {moments[0]:.5f}, "
        f"Standard Deviation = {moments[1]:.5f}, "
        f"Skewness = {moments[2]:.5f}, "
        f"Excess Kurtosis = {moments[3]:.5f}."
    )

    if moments[2] > 0:
        skew_text = "right skewed"
    elif moments[2] < 0:
        skew_text = "left skewed"
    else:
        skew_text = "not skewed"

    if moments[3] > 0:
        kurt_text = "leptokurtic"
    elif moments[3] < 0:
        kurt_text = "platykurtic"
    else:
        kurt_text = "mesokurtic"

    print(f"The data is {skew_text} and {kurt_text}.")


def main():
    """Main execution."""
    df = pd.read_csv("crypto50_combined.csv", parse_dates=["Date"])

    df = preprocessing(df)

    col = "LogReturn"

    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == "__main__":
    main()

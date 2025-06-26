import pandas as pd
import numpy as np
from dta_handling import load_data

def assign_typology(df):
    """
    Assigns wealth and income percentiles as numeric ranks and classifies mismatch typologies.
    Conditions:
    - Wealth-rich = top 25% (>= 75th percentile)
    - Wealth-poor = bottom 50% (< 50th percentile)
    - Income-rich = top 20% (>= 80th percentile)
    - Income-poor = bottom 60% (< 60th percentile)
    """

    df = df.copy()
    df["wealth_percentile"] = df["riquezanet"].rank(pct=True)
    df["income_percentile"] = df["renthog21_eur22"].rank(pct=True)

    def classify(row):
        w = row["wealth_percentile"]
        i = row["income_percentile"]

        if w >= 0.75 and i < 0.60:
            return "Wealth-rich, income-poor"
        elif w < 0.50 and i >= 0.80:
            return "Income-rich, wealth-poor"
        else:
            return "Aligned"

    df["mismatch_type"] = df.apply(classify, axis=1)
    return df


def get_typology_statistics(df):
    typology_counts = df["mismatch_type"].value_counts().reset_index()
    typology_counts.columns = ["mismatch_type", "count"]
    print("\nCounts of each mismatch type:")
    print(typology_counts)

    shares = df.groupby("mismatch_type")["facine3"].sum() / df["facine3"].sum()
    print("\nWeighted population share by mismatch type:")
    print(shares)

    summary_stats = df.groupby("mismatch_type").apply(
        lambda x: pd.Series(
            {
                "mean_wealth": np.average(x["riquezanet"], weights=x["facine3"]),
                "mean_income": np.average(x["renthog21_eur22"], weights=x["facine3"]),
            }
        )
    )
    print("\nAverage wealth and income by mismatch type:")
    print(summary_stats)

    return typology_counts, shares, summary_stats

def get_typology_wealth_percentile(df):
    """
    Assigns wealth bins using predefined percentile intervals
    and calculates the weighted distribution of mismatch types across them.
    """

    df = df.copy()
    labels = {
        1: "< P20",
        2: "P20–P40",
        3: "P40–P60",
        4: "P60–P80",
        5: "P80–P90",
        6: "> P90"
    }

    df["wealth_bin_code"] = pd.cut(
        df["wealth_percentile"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
        labels=[1, 2, 3, 4, 5, 6],
        include_lowest=True
    ).astype(int)

    df["wealth_bin"] = df["wealth_bin_code"].map(labels)

    table = (
        df.groupby(["mismatch_type", "wealth_bin"])["facine3"]
        .sum()
        .groupby(level=0)
        .apply(lambda x: 100 * x / x.sum())
        .unstack()
        .fillna(0)
        .round(2)
    )

    print("\nDistribution of mismatch types across custom wealth bins (% within typology):")
    print(table)


    return table


def income_distribution_by_wealth(df):
    """
    Create a weighted cross-tab of income bins within each wealth bin (percrent vs. percriq)
    """

    df["percriq"] = df["percriq"].astype(str)
    df["percrent"] = df["percrent"].astype(str)

    cross_tab = (
        df.groupby(["percrent", "percriq"])["facine3"]
        .sum()
        .groupby(level=0)
        .apply(lambda x: 100 * x / x.sum()) 
        .unstack()
        .fillna(0)
        .round(2)
    )

    print("\nIncome class distribution within each wealth bin (%):")
    print(cross_tab)

    return cross_tab

import seaborn as sns
import matplotlib.pyplot as plt

def plot_income_distribution_heatmap(cross_tab):
    """
    Plot heatmap of income distribution across wealth bins.
    """
    plt.figure(figsize=(10, 5))
    sns.heatmap(
        cross_tab,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=.5,
        cbar_kws={"label": "% of Households"},
    )

    plt.title("Income Distribution Within Wealth Percentile Groups")
    plt.xlabel("Income Percentile Bin")
    plt.ylabel("Wealth Percentile Bin")
    plt.tight_layout()
    plt.savefig("income_within_wealth_bins_heatmap.png", dpi=300)
    plt.show()


def main():
    df = load_data() 
    df = assign_typology(df)

    get_typology_statistics(df)
    get_typology_wealth_percentile(df)
    cross_tab = income_distribution_by_wealth(df)
    plot_income_distribution_heatmap(cross_tab)

    return df




if __name__ == "__main__":
    main()

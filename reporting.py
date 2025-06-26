import numpy as np
import pandas as pd
from ineqpy.inequality import gini

from constants import Net_Wealth
from statistic import top_share

USE_INDIVIDUAL = False  # Set to False to run household-level model

wealth_col = "netwealth_individual" if USE_INDIVIDUAL else Net_Wealth

def summarize_cap_and_tax_shares(df):
    """
    Summarizes cap relief and final tax shares for top wealth percentiles,
    averaging results over implicates (following EFF methodology).
    """
    results = []

    for imp, sub_df in df.groupby("imputation"):
        top_10 = sub_df["wealth_rank"] > 0.90
        top_1 = sub_df["wealth_rank"] > 0.99
        top_01 = sub_df["wealth_rank"] > 0.999

        total_relief = (sub_df["cap_relief"] * sub_df["facine3"]).sum()
        total_final_tax = (sub_df["adjusted_final_tax"] * sub_df["facine3"]).sum()

        top10_relief = (sub_df.loc[top_10, "cap_relief"] * sub_df.loc[top_10, "facine3"]).sum()
        top1_relief = (sub_df.loc[top_1, "cap_relief"] * sub_df.loc[top_1, "facine3"]).sum()
        top01_relief = (sub_df.loc[top_01, "cap_relief"] * sub_df.loc[top_01, "facine3"]).sum()

        top10_tax = (sub_df.loc[top_10, "adjusted_final_tax"] * sub_df.loc[top_10, "facine3"]).sum()
        top1_tax = (sub_df.loc[top_1, "adjusted_final_tax"] * sub_df.loc[top_1, "facine3"]).sum()
        top01_tax = (sub_df.loc[top_01, "adjusted_final_tax"] * sub_df.loc[top_01, "facine3"]).sum()

        results.append({
            "Cap_Top10": top10_relief / total_relief if total_relief > 0 else np.nan,
            "Cap_Top1": top1_relief / total_relief if total_relief > 0 else np.nan,
            "Cap_Top01": top01_relief / total_relief if total_relief > 0 else np.nan,
            "Tax_Top10": top10_tax / total_final_tax if total_final_tax > 0 else np.nan,
            "Tax_Top1": top1_tax / total_final_tax if total_final_tax > 0 else np.nan,
            "Tax_Top01": top01_tax / total_final_tax if total_final_tax > 0 else np.nan,
        })

    summary_df = pd.DataFrame(results)
    mean_shares = summary_df.mean()

    print("Cap Relief Share (averaged over implicates):")
    print(f"  Top 10%: {mean_shares['Cap_Top10']:.2%}")
    print(f"  Top 1%:  {mean_shares['Cap_Top1']:.2%}")
    print(f"  Top 0.1%: {mean_shares['Cap_Top01']:.2%}\n")

    print("Final Tax Share (averaged over implicates):")
    print(f"  Top 10%: {mean_shares['Tax_Top10']:.2%}")
    print(f"  Top 1%:  {mean_shares['Tax_Top1']:.2%}")
    print(f"  Top 0.1%: {mean_shares['Tax_Top01']:.2%}")

def report_effective_tax_rates(df):
    def weighted_avg(series, weights):
        mask = series.notna()
        return np.average(series[mask], weights=weights[mask])

    results = []

    for imp, sub_df in df.groupby("imputation"):
        top_10 = sub_df["wealth_rank"] > 0.90
        top_1 = sub_df["wealth_rank"] > 0.99

        eff_tax_top10 = weighted_avg(
            sub_df.loc[top_10, "eff_tax_rate"], sub_df.loc[top_10, "facine3"]
        )
        eff_tax_top1 = weighted_avg(
            sub_df.loc[top_1, "eff_tax_rate"], sub_df.loc[top_1, "facine3"]
        )
        eff_tax_top10_nocap = weighted_avg(
            sub_df.loc[top_10, "eff_tax_nocap"], sub_df.loc[top_10, "facine3"]
        )
        eff_tax_top1_nocap = weighted_avg(
            sub_df.loc[top_1, "eff_tax_nocap"], sub_df.loc[top_1, "facine3"]
        )

        results.append({
            "cap_10": eff_tax_top10,
            "cap_1": eff_tax_top1,
            "nocap_10": eff_tax_top10_nocap,
            "nocap_1": eff_tax_top1_nocap,
        })

    summary = pd.DataFrame(results).mean()

    print("\n--- Effective Tax Rates (averaged over implicates) ---")
    print(f"With Cap - Top 10%: {summary['cap_10']:.3%}")
    print(f"With Cap - Top 1%:  {summary['cap_1']:.3%}")
    print(f"Without Cap - Top 10%: {summary['nocap_10']:.3%}")
    print(f"Without Cap - Top 1%:  {summary['nocap_1']:.3%}")

    return df

def typology_impact_summary(df, weight_col="facine3"):

    grouped = df.groupby(["imputation", "mismatch_type"])

    result = grouped.apply(
        lambda g: pd.Series({
            "Population Share": g[weight_col].sum(),
            "Avg Final Tax": np.average(g["adjusted_final_tax"], weights=g[weight_col]),
            "Cap Relief Share": np.average((g["cap_relief"] > 1e-6), weights=g[weight_col]),
            "Migration Rate": np.average(g["Migration_Exit"], weights=g[weight_col]),
            "Revenue per Capita": np.average(g["adjusted_final_tax"], weights=g[weight_col])
 })
    ).reset_index()

    total_pop_weight = result.groupby("imputation")["Population Share"].transform("sum")
    result["Population Share"] /= total_pop_weight

    final = result.groupby("mismatch_type").mean().reset_index()

    print("\n--- Typology Impact Table (averaged over implicates) ---")
    print(final.to_string(index=False))
    return final


import numpy as np

def gini(values, weights=None):
    """
    Compute Gini coefficient of a numpy array or pandas Series.

    Parameters:
    - values: array-like, income or wealth values
    - weights: array-like, same length as values

    Returns:
    - Gini coefficient as float between 0 and 1
    """
    values = np.asarray(values)
    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights)

    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]

    cumw = np.cumsum(sorted_weights)
    cumxw = np.cumsum(sorted_values * sorted_weights)

    gini_numerator = np.sum(sorted_weights * (cumxw - sorted_values * sorted_weights / 2))
    gini_denominator = cumxw[-1] * cumw[-1]
    
    return 1 - 2 * gini_numerator / gini_denominator
def compute_inequality_metrics(df, weight_col="facine3"):
    implicate_metrics = []

    for imp, group in df.groupby("imputation"):
        result = {
            "Gini Before Tax": gini(group[wealth_col], weights=group[weight_col]),
            "Gini After Tax (cap)": gini(group["wealth_after_cap"], weights=group[weight_col]),
            "Gini After Tax (no cap)": gini(group["wealth_after_no_cap"], weights=group[weight_col]),

            "Top 10% Share Before": top_share(group, wealth_col, weight_col, 0.10),
            "Top 10% Share After (cap)": top_share(group, "wealth_after_cap", weight_col, 0.10),
            "Top 10% Share After (no cap)": top_share(group, "wealth_after_no_cap", weight_col, 0.10),

            "Top 1% Share Before": top_share(group, wealth_col, weight_col, 0.01),
            "Top 1% Share After (cap)": top_share(group, "wealth_after_cap", weight_col, 0.01),
            "Top 1% Share After (no cap)": top_share(group, "wealth_after_no_cap", weight_col, 0.01),
        }
        implicate_metrics.append(result)

    metric_df = pd.DataFrame(implicate_metrics)
    avg_metrics = metric_df.mean().to_dict()

    print("\n--- Inequality Metrics (averaged over implicates) ---")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4%}")

    return avg_metrics


def payer_coverage(df):
    payers = (df["final_tax"] > 0).mean()
    print(f"Coverage: {payers:.2%} of population pays any WT.")

def generate_summary_table2(df: pd.DataFrame, weight_col="facine3") -> None:
    """
    Generate and print summary of tax revenue at different simulation stages,
    averaged over implicates.
    """
    def revenue_components(g):
        w = g[weight_col]
        return pd.Series({
            "Revenue Before Erosion": (g["sim_tax_original"] * w).sum(),
            "Revenue After Behavioral Erosion": (g["tax_afterBR"] * w).sum(),
            "Revenue After Income Cap": (g["final_tax"] * w).sum(),
            "Revenue After Regional Adjustments": (g["adjusted_final_tax"] * w).sum()
        })

    summary_by_imp = df.groupby("imputation").apply(revenue_components)
    summary = summary_by_imp.mean()

    # Compute losses
    erosion_loss = summary["Revenue Before Erosion"] - summary["Revenue After Behavioral Erosion"]
    cap_relief_loss = summary["Revenue After Behavioral Erosion"] - summary["Revenue After Income Cap"]
    regional_loss = summary["Revenue After Income Cap"] - summary["Revenue After Regional Adjustments"]

    print("\n--- Revenue Summary (averaged over implicates) ---")
    print(f"Revenue Before Erosion:            €{summary['Revenue Before Erosion']:,.0f}")
    print(f"Revenue After Behavioral Erosion:  €{summary['Revenue After Behavioral Erosion']:,.0f}")
    print(f"Revenue After Income Cap:          €{summary['Revenue After Income Cap']:,.0f}")
    print(f"Revenue After Regional Adjustments:€{summary['Revenue After Regional Adjustments']:,.0f}")
    print(f"\nLoss Due to Behavioral Erosion:    €{erosion_loss:,.0f}")
    print(f"Loss Due to Income Cap Relief:     €{cap_relief_loss:,.0f}")
    print(f"Loss Due to Regional Adjustments:  €{regional_loss:,.0f}")


def loss_breakdown(df: pd.DataFrame, weight_col="facine3") -> None:
    """
    Breakdown of tax revenue losses by type, averaged across implicates.
    """
    def losses(g):
        w = g[weight_col]
        gross = (g["sim_tax_original"] * w).sum()
        cap_loss = ((g["tax_afterBR"] - g["final_tax"]) * w).sum()
        regional_loss = ((g["final_tax"] - g["adjusted_final_tax"]) * w).sum()
        behav_loss = ((g["sim_tax_original"] - g["tax_afterBR"]) * w).sum()
        return pd.Series({
            "Gross": gross,
            "Cap Loss": cap_loss,
            "Regional Loss": regional_loss,
            "Behavioral Loss": behav_loss
        })

    loss_df = df.groupby("imputation").apply(losses)
    avg = loss_df.mean()

    print("\n--- Loss Breakdown (averaged over implicates) ---")
    print(f"Cap loss:      {avg['Cap Loss'] / avg['Gross']:.1%} of gross")
    print(f"Regional loss: {avg['Regional Loss'] / avg['Gross']:.1%}")
    print(f"Behavioural:   {avg['Behavioral Loss'] / avg['Gross']:.1%}")

import matplotlib.pyplot as plt

def plot_revenue_decomposition(df: pd.DataFrame, weight_col="facine3") -> None:
    plt.style.use('bmh') 
    def revenue_components(g):
        w = g[weight_col]
        return pd.Series({
            "Before Erosion": (g["sim_tax_original"] * w).sum(),
            "After Behavior": (g["tax_afterBR"] * w).sum(),
            "After Cap": (g["final_tax"] * w).sum(),
            "After Regional": (g["adjusted_final_tax"] * w).sum()
        })

    summary = df.groupby("imputation").apply(revenue_components).mean()

    stages = ["Before Erosion", "After Behavior", "After Cap", "After Regional"]
    values = [summary[stage] / 1e9 for stage in stages]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(stages, values, color=["#5f8fef", "#8cd3f7e7", "#80edd6d2", "#a9e7c4d1"])

    ax.set_ylabel("Revenue (€ billions)")
    ax.set_title("Breakdown of Wealth Tax Revenue by Erosion Stage", fontsize=14)
    ax.set_ylim(0, max(values) * 1.1)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{height:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

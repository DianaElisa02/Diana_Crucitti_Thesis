import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from constants import (
    PROGRESSIVE_TAX_BRACKETS,
    wealth_percentile,
    Net_Wealth,
    Income,
    Primary_Residence,
    Business_Value,
    Residence_Ownership,
    Business_Ownership,
    Num_Workers,
    SPANISH_PIT_2022_BRACKETS,
    
)
from dta_handling import load_data
from eff_typology import assign_typology
from preprocessing import individual_split
from reporting import (
    summarize_cap_and_tax_shares,
    report_effective_tax_rates,
    typology_impact_summary,
    generate_summary_table2,
    compute_inequality_metrics,
    payer_coverage,
    loss_breakdown,
    plot_revenue_decomposition,
)
from wealth_tax import simulate_household_wealth_tax


# Set simulation parameters for sensitivity analysis, when setting true is a sensitivity analysis
USE_INDIVIDUAL = False  # Set to False to run household-level model, or set to True for individual-level(sensitivity analysis)
LOWER_BEHAVIOURAL_RESPONSE = False
NO_REGIONAL_EROSION = False
wealth_col = "netwealth_individual" if USE_INDIVIDUAL else Net_Wealth
income_col = "income_individual" if USE_INDIVIDUAL else Income


def compute_legal_exemptions(df):
    """s
    Estimates total legal exemptions that can be subtracted from taxable wealth.

    Two main categories are considered:
    - Primary residence exemption (if owned)
    - Business asset exemption (applied probabilistically)

    The idea is to replicate legal treatments where exemptions reduce the tax base
    before applying any tax rates.
    """

    owns_home = df[Residence_Ownership] == "Ownership"
    primary_home_val = df[Primary_Residence].fillna(0)
    exempt_home_value = np.where(owns_home, np.minimum(primary_home_val, 300_000), 0)

    # Business exemption if household has declared business value
    business_exemption_rate = 0.30  # Based on literature(Duran-Cabré et al. 2021)
    has_business_value = df[Business_Ownership] == 1
    apply_business_exempt = (
        np.random.rand(len(df)) < business_exemption_rate
    ) & has_business_value
    business_exempt = np.where(apply_business_exempt, df[Business_Value].fillna(0), 0)

    return exempt_home_value + business_exempt

def simulate_pit_liability(df: pd.DataFrame, correction_top1=0.15, weight_col="facine3"):
    """
    Simulates Spanish PIT liability with a basic personal allowance.
    Also applies an upward correction to the top 1% to approximate unreported capital income.

    Parameters:

    - weight_col: name of weight column (default: 'facine3')
    """
    df = df.copy()

    personal_allowance = 5550
    taxable_income = np.maximum(df[income_col] - personal_allowance, 0)

    df["pit_liability"] = taxable_income.apply(
        lambda amount: calculate_tax_liability(amount, SPANISH_PIT_2022_BRACKETS)
    )

    total_pit = (df["pit_liability"] * df[weight_col]).sum()

    print(f"Total PIT (before correction):  €{total_pit:,.2f}")
    return df


def apply_wealth_tax_income_cap(
    df: pd.DataFrame, income_cap_rate: float = 0.60, min_wealth_tax_share: float = 0.20
):
    """
    Apply an income-based cap to the wealth tax (WT) as per Spanish tax rules.

    Ensures that the total tax burden (PIT + WT) does not exceed a set percentage
    (e.g. 60%) of an individual's income. If it does, the WT is reduced—but not
    below a minimum share (e.g. 20%) of the original wealth tax.

    Parameters:
    - income_cap_rate: ceiling threshold (default = 60%)
    - min_wt_share: minimum WT share to preserve (default = 20%)

    Returns:
    - df: DataFrame with capped WT and relief columns
    """
    df = df.copy()
    eligible = df[wealth_col] < 90_000_000
    income_limit = df[income_col] * income_cap_rate
    wealth_tax = df["tax_afterBR"].fillna(0)
    income_tax = df["pit_liability"].fillna(0)


    total_tax = wealth_tax + income_tax
    over_cap = (total_tax > income_limit)

    max_allowed_relief = wealth_tax * (1 - min_wealth_tax_share)

    excess = total_tax - income_limit
    wt_relief = np.minimum(excess, max_allowed_relief)
    wt_relief = np.where(over_cap, wt_relief, 0.0)

    df["cap_relief"] = wt_relief
    df["final_tax"] = wealth_tax - wt_relief
    wealth_tax = df["tax_afterBR"].fillna(0)


    return df


def calculate_tax_liability(
    amount: float, brackets: list[tuple[float, float, float]]
) -> float:
    """
    Compute total tax liability using progressive brackets.
    """
    return sum(
        max(0, min(amount, upper_limit) - lower_limit) * rate
        for lower_limit, upper_limit, rate in brackets
    )


def simulate_household_wealth_tax(
    df: pd.DataFrame, exemption_amount: int = 700_000
) -> pd.DataFrame:
    """
    Simulate a progressive wealth tax based on individual net wealth,
    taking into account legal exemptions and non-taxable assets.

    Returns:
    -------
    pd.DataFrame
        Original DataFrame with added columns:
            - exempt_total: legal exemption calculated for each individual.
            - taxable_wealth: wealth subject to tax after exemptions.
            - sim_tax: simulated tax owed under a progressive tax system.
    """

    df = df.copy()

    df["exempt_total"] = compute_legal_exemptions(df)

    # Non-taxable assets: art and vehicles
    non_taxable_assets = (
        df["p2_71"].fillna(0) + df["timpvehic"].fillna(0) + df["p2_84"].fillna(0)
    )

    # Taxable wealth = net wealth - non-taxable assets - legal exemptions - base exemption
    adjusted_wealth = (
        df[wealth_col] - non_taxable_assets - df["exempt_total"]
    )
    df["taxable_wealth"] = np.maximum(adjusted_wealth - exemption_amount, 0)

    df["sim_tax"] = df["taxable_wealth"].apply(
        lambda amount: calculate_tax_liability(amount, PROGRESSIVE_TAX_BRACKETS)
    )
    df["sim_tax_original"] = df["sim_tax"] 

    return df

def get_top_marginal_tax(amount, brackets):
    for lower, upper, rate in reversed(brackets):
        if amount > lower:
            return rate
    return 0.0

def recalculate_wealth_tax_on_eroded_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute sim_tax using taxable_wealth_eroded instead of the original base.
    This ensures behavioral erosion actually reduces the tax owed.
    """
    df = df.copy()
    df["tax_afterBR"] = df["taxable_wealth_eroded"].apply(
        lambda amount: calculate_tax_liability(amount, PROGRESSIVE_TAX_BRACKETS)
    )
    return df

def apply_behavioral_response(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies behavioural erosion to taxable wealth based on Jakobsen et al. (2020) and Brülhart et al. (2022).
    Uses elasticities estimated for the very wealthy and applies erosion only to top 10% households.

    Erosion = elasticity × marginal tax rate × taxable wealth
    """

    df = df.copy()

    # Assign elasticity based on wealth rank
    def assign_elasticity(rank):
        if rank >= 0.9999:
            return 10.0  # Jakobsen: top 0.01%
        elif rank >= 0.999:
            return 7.0   # Jakobsen: top 0.1%
        elif rank >= 0.99:
            return 2.0   # Brülhart-like elasticity
        elif rank >= 0.90:
            return 0.5   # Minor response
        else:
            return 0.0

    df["behavioral_elasticity"] = df["wealth_rank"].apply(assign_elasticity)


    df["top_marginal_rate"] = df["taxable_wealth"].apply(
        lambda x: get_top_marginal_tax(x, PROGRESSIVE_TAX_BRACKETS)
    )

    erosion_fraction = df["behavioral_elasticity"] * df["top_marginal_rate"]
    erosion_fraction = erosion_fraction.clip(upper=0.70) 

    df["erosion_factor"] = 1 - erosion_fraction
    df["taxable_wealth_eroded"] = df["taxable_wealth"] * df["erosion_factor"]
    df["%_eroded"] = 100 * (1 - df["erosion_factor"])

    return df
 
def simulate_migration_attrition(
    df: pd.DataFrame,
    wealth_threshold: float = 0.999,
    base_migration_prob: float = 0.02,
    elasticity: float = 1.76,
) -> pd.DataFrame:
    """
    Simulates tax-motivated migration or wealth erosion among top wealth holders,
    based on behavioral responses modeled in Jakobsen et al. (2020).

    This function probabilistically identifies individuals likely to "exit"
    the tax base (e.g., through migration, legal restructuring, or non-compliance)
    as a function of their effective wealth tax burden.

    Parameters:
    - top_pct (float): threshold above which individuals are considered part of the top wealth group (default: 99.8th percentile)
    - base_prob (float): baseline probability of migration at zero tax (default: 4%)
    - elasticity (float): behavioral response elasticity of migration to net-of-tax rate

    Returns:
    - df (DataFrame): updated DataFrame with migration exit flags and adjusted tax contributions
    """
    df = df.copy()
    df["Migration_Exit"] = False

    net_of_tax = 1 - df["final_tax"] / (df[wealth_col] + 1e-6)

    exit_prob = base_migration_prob * np.exp(elasticity * (1 - net_of_tax))

    top_wealth_group = df["wealth_rank"] > wealth_threshold
    will_migrate = (np.random.rand(len(df)) < exit_prob) & top_wealth_group

    df.loc[will_migrate, "Migration_Exit"] = True
    df.loc[will_migrate, ["sim_tax_original", "tax_afterBR", "final_tax", "taxable_wealth_eroded"]] = 0


    return df

def apply_regional_tax_adjustments(
    df: pd.DataFrame, tax_reduction: float = 0.1
) -> pd.DataFrame:
    """Adjust taxable wealth and tax values to account for regional exemptions such as Andalusia
    """
    df = df.copy()
    adjustment_factor = 1 - tax_reduction

    df["adjusted_sim_tax_original"] = df["sim_tax_original"] * adjustment_factor
    df["adjusted_tax_afterBR"] = df["tax_afterBR"] * adjustment_factor
    df["adjusted_final_tax"] = df["final_tax"] * adjustment_factor

    return df

def compute_effective_tax_rates(df):
    df = df.copy()
    df["eff_tax_rate"] = df["adjusted_final_tax"] / (df[wealth_col] + 1e-6)
    df["eff_tax_rate"] = df["eff_tax_rate"].replace([np.inf, -np.inf], np.nan)

    df["eff_tax_nocap"] = df["adjusted_tax_afterBR"] / (df[wealth_col] + 1e-6)
    df["eff_tax_nocap"] = df["eff_tax_nocap"].replace([np.inf, -np.inf], np.nan)
    return df


def compute_net_wealth_post_tax(df):
    df = df.copy()
    df["wealth_after_cap"] = df[wealth_col] - df[
        "adjusted_final_tax"
    ].fillna(0)
    df["wealth_after_no_cap"] = df[wealth_col] - df[
        "adjusted_tax_afterBR"
    ].fillna(0)
    return df


def check_valid_input_data(df):
    assert not (df[wealth_col].isna()).any()

def compute_weighted_wealth_rank(df, wealth_col=wealth_col, weight_col="facine3"):
    df = df.copy()
    result = []

    for imp in df["imputation"].unique():
        sub_df = df[df["imputation"] == imp].copy()

        total_weight = sub_df[weight_col].sum()
        sub_df["scaled_weight"] = sub_df[weight_col] / total_weight

        sub_df = sub_df.sort_values(by=wealth_col, kind="mergesort").reset_index(drop=True)
        sub_df["cum_weight"] = sub_df["scaled_weight"].cumsum()

        sub_df["wealth_rank"] = sub_df["cum_weight"]

        result.append(sub_df)

    df_ranked = pd.concat(result, ignore_index=True)
    return df_ranked

import matplotlib.pyplot as plt
import seaborn as sns

def plot_wealth_distribution(df):
    """
    Plots average wealth by predefined wealth percentile group (percrent).
    """
    grouped = df.groupby("percrent").apply(
        lambda x: pd.Series({
            "mean_wealth": np.average(x["riquezanet"], weights=x["facine3"]),
            "population_share": x["facine3"].sum() / df["facine3"].sum()
        })
    ).reset_index()

    bin_order = ["< P20", "P20-P40", "P40-P60", "P60-80", "P80-P90", "> P90"]
    grouped["percrent"] = pd.Categorical(grouped["percrent"], categories=bin_order, ordered=True)
    grouped = grouped.sort_values("percrent")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    sns.barplot(data=grouped, x="percrent", y="mean_wealth", ax=ax1, color="#5DA5DA")
    ax1.set_ylabel("Mean Net Wealth (€)", fontsize=12)
    ax1.set_xlabel("Wealth Percentile Group", fontsize=12)
    ax1.set_title("Wealth Distribution by Wealth Percentile Group", fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Optional: Annotate bars
    for i, row in grouped.iterrows():
        ax1.text(i, row["mean_wealth"] + 5000, f"€{row['mean_wealth']:,.0f}", 
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

def main():
    np.random.seed(42)

    df = load_data()


    df = individual_split(df)
    
    check_valid_input_data(df)

    wealth_col = "netwealth_individual" if USE_INDIVIDUAL else Net_Wealth
    income_col = "income_individual" if USE_INDIVIDUAL else Income

    df = assign_typology(df)
    df = compute_weighted_wealth_rank(df, wealth_col, "facine3")

    df = simulate_household_wealth_tax(df, exemption_amount=700_000)
    df["sim_tax_original"] = df["sim_tax"]
    
    df["top_marginal_rate"] = df["taxable_wealth"].apply(
    lambda x: get_top_marginal_tax(x, PROGRESSIVE_TAX_BRACKETS)
)


    df = apply_behavioral_response(df)
    
    if USE_INDIVIDUAL:
        def assign_elasticity(net_w, rank=None):
            if LOWER_BEHAVIOURAL_RESPONSE:
                # Downscaled elasticities
                if net_w >= 10_000_000:
                    return 0.8
                elif net_w >= 5_000_000:
                    return 0.5
                elif net_w >= 1_000_000:
                    return 0.2
                else:
                    return 0.0
            else:
                if rank is None:
                    return 0.0
                if rank >= 0.9999:
                    return 10.0  # Jakobsen: top 0.01%
                elif rank >= 0.999:
                    return 7.0   # Jakobsen: top 0.1%
                elif rank >= 0.99:
                    return 2.0   # Brülhart-like elasticity
                elif rank >= 0.90:
                    return 0.5   # Minor response
                else:
                    return 0.0

        df["behavioral_elasticity"] = df.apply(
            lambda row: assign_elasticity(row["netwealth_individual"], row.get("wealth_rank", None)),
            axis=1
        )


    df = recalculate_wealth_tax_on_eroded_base(df)
    df = simulate_pit_liability(df)
    df = apply_wealth_tax_income_cap(df)
    df = simulate_migration_attrition(df)
    print(df["Migration_Exit"].value_counts())
    if not NO_REGIONAL_EROSION:
        df = apply_regional_tax_adjustments(df)
    else:
        df["adjusted_sim_tax_original"] = df["sim_tax_original"]
        df["adjusted_tax_afterBR"] = df["tax_afterBR"]
        df["adjusted_final_tax"] = df["final_tax"]


    generate_summary_table2(df)
    typology_impact_summary(df)

    df = compute_effective_tax_rates(df)
    report_effective_tax_rates(df)
    summarize_cap_and_tax_shares(df)

    df["wealth_after_cap"] = df[wealth_col] - df["final_tax"].fillna(0)
    df["wealth_after_no_cap"] = df[wealth_col] - df["tax_afterBR"].fillna(0)

    df = compute_net_wealth_post_tax(df)

    compute_inequality_metrics(df)
    payer_coverage(df)
    loss_breakdown(df)

    relieved = df["cap_relief"] > 0
    share_relieved = (df[relieved]["facine3"].sum() / df["facine3"].sum()) * 100
    avg_relief = (df[relieved]["cap_relief"] * df[relieved]["facine3"]).sum() / df["facine3"].sum()

    print(f"% of population receiving relief: {share_relieved:.2f}%")
    print(f"Average relief per capita (weighted): €{avg_relief:,.2f}")

    df["weighted_relief"] = df["cap_relief"] * df["facine3"]

    relief_by_typology = (
    df.groupby("mismatch_type")["weighted_relief"].sum()
    / (df["cap_relief"] * df["facine3"]).sum()
)

    print("Share of total cap relief received by typology:")
    print(relief_by_typology)

    #df = plot_revenue_decomposition(df)




if __name__ == "__main__":
    main()

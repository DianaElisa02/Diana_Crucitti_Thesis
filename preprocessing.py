from constants import Num_Workers, Net_Wealth, Income, Primary_Residence, Business_Value
from dta_handling import load_data
import pandas as pd


def individual_split(df):
    """
    Decomposes household-level net wealth and income into individual-level equivalents.

    Since income and wealth are reported per household, this function tries to approximate
    per-capita figures by dividing by the number of economic contributors (working adults).
    Where no earners are reported, one worker is assumed as a fallback proxy.
    """
    df = df.copy()
    adult_split_factor = df[Num_Workers].clip(lower=1)

    df["netwealth_individual"] = df[Net_Wealth] / adult_split_factor
    df["income_individual"] = df[Income] / adult_split_factor

    return df


def apply_valuation_manipulation(df, real_estate_discount=0.15, business_discount=0.20):
    """
    Adjusts reported asset values for typical underreporting in household surveys.

    Applies empirical discounts to real estate and business holdings, in line with literature
    showing systematic undervaluation in self-reported data.

    References:
      - Alstadsæter et al. (2019), AER
      - Advani & Tarrant (2022), IFS
      - Duran-Cabré et al. (2023)

    Parameters:
    - real_estate_discount: fraction to reduce real estate values by (default: 15%)
    - business_discount: fraction to reduce business asset values by (default: 20%)
    """
    df = df.copy()
    df[Primary_Residence] = df[Primary_Residence] * (1 - real_estate_discount)
    df[Business_Value] = df[Business_Value] * (1 - business_discount)
    return df


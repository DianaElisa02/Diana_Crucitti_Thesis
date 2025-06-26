from __future__ import annotations
from typing import Sequence

import numpy as np
import pandas as pd

from constants import (
    NON_TAXABLE_ASSET_COLS,
    Num_Workers,
    PROGRESSIVE_TAX_BRACKETS,
    EXEMPT_TOTAL_COL,
    NET_WEALTH_COL,
    TAXABLE_WEALTH_COL,
    SIM_TAX_COL,
    Business_Ownership,
    Primary_Residence,
    Residence_Ownership,
    Business_Value,
    SPANISH_PIT_2022_BRACKETS,
    INCOME_COL,
    PIT_LIABILITY_COL,
)


def progressive_tax(series: pd.Series, brackets: Sequence[tuple]) -> pd.Series:
    """
    Vectorised marginal-rate tax calculator.
    `series` can be a Pandas Series or NumPy array.
    """
    tax = np.zeros_like(series, dtype=float)

    for lower, upper, rate in brackets:
        in_bracket = series.clip(lower, upper) - lower
        in_bracket[in_bracket < 0] = 0
        tax += in_bracket * rate
    return pd.Series(tax, index=series.index, name="tax")


def _non_taxable_assets(
    df: pd.DataFrame,
    asset_cols: Sequence[str] = NON_TAXABLE_ASSET_COLS,
    earners_col: str = Num_Workers,
) -> pd.Series:
    """Return a Series with per-adult non-taxable asset value.

    If there are no workers in the household, we assume 1"""
    earners = df[earners_col].clip(lower=1)
    assets = df[list(asset_cols)].fillna(0).sum(axis=1)
    return assets / earners


def simulate_household_wealth_tax(
    df: pd.DataFrame,
    exemption_amount: int = 700_000,
    brackets: Sequence[tuple] = PROGRESSIVE_TAX_BRACKETS,
    asset_cols: Sequence[str] = NON_TAXABLE_ASSET_COLS,
) -> pd.DataFrame:
    """
    Compute legal exemptions ➜ taxable base ➜ wealth-tax liability

    Adds / overwrites three columns:
      * exempt_total
      * taxable_wealth
      * sim_tax
    """
    out = df.copy()

    out[EXEMPT_TOTAL_COL] = compute_legal_exemptions(out)
    nontax = _non_taxable_assets(out, asset_cols)
    adjusted = out[NET_WEALTH_COL] - nontax - out[EXEMPT_TOTAL_COL] - exemption_amount
    out[TAXABLE_WEALTH_COL] = adjusted.clip(lower=0)

    out[SIM_TAX_COL] = progressive_tax(out[TAXABLE_WEALTH_COL], brackets)

    return out


def compute_legal_exemptions(df):
    """
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


def simulate_pit_liability2(
    df: pd.DataFrame,
    brackets: list[tuple[float, float, float]] = SPANISH_PIT_2022_BRACKETS,
    income_col: str = INCOME_COL,
) -> pd.DataFrame:
    """
    Vectorised Spanish PIT simulation (no deductions/exemptions).

    Adds/overwrites column:  *pit_liability*
    """
    out = df.copy()
    out[PIT_LIABILITY_COL] = progressive_tax(out[income_col], brackets)
    return out

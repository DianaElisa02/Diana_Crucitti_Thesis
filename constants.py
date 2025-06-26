# Dataframe column names
Net_Wealth = "riquezanet"
PEOPLE_IN_HOUSEHOLD = "np1"
Income = "renthog21_eur22"
wealth_percentile = "percriq"
working_status = "nsitlabdom"
income_percentile = "percrent"
Num_Workers = "nnumadtrab"
Primary_Residence = "p2_70"
Business_Value = "valhog"
Residence_Ownership = "np2_1"
Business_Ownership = "havenegval"
NON_TAXABLE_ASSET_COLS: tuple[str, ...] = ("p2_71", "timpvehic", "p2_84")
NET_WEALTH_COL = "netwealth_individual"
EXEMPT_TOTAL_COL = "exempt_total"
TAXABLE_WEALTH_COL = "taxable_wealth"
SIM_TAX_COL = "sim_tax"
INCOME_COL = "income_individual"
PIT_LIABILITY_COL = "pit_liability"

SPANISH_PIT_2022_BRACKETS = [
    (0, 12_450, 0.19),
    (12_450.01, 20_200, 0.24),
    (20_200.01, 35_200, 0.30),
    (35_200.01, 60_000, 0.37),
    (60_000.01, 300_000, 0.45),
    (300_000.01, float("inf"), 0.47),
]

PROGRESSIVE_TAX_BRACKETS = [
    (0, 167129.45, 0.002),
    (167129.46, 334246.88, 0.003),
    (334246.89, 668499.75, 0.005),
    (668499.76, 1336999.51, 0.009),
    (1336999.52, 2673999.01, 0.013),
    (2673999.02, 5347998.03, 0.017),
    (5347998.04, 10695996.06, 0.021),
    (10695996.07, float("inf"), 0.035),
]



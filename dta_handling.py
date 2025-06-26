import numpy as np
import pandas as pd
from pandas import unique

from constants import PEOPLE_IN_HOUSEHOLD, Num_Workers


def check_nan(df: pd.DataFrame, column: str) -> None:
    print(f"Unique values in column {column}: {unique(df[column])}")

    rows = df[df[column].isnull()]
    print(rows)
    rows = df[df[column] == pd.NA]
    print(rows)

    rows = df[df[column] == np.nan]
    print(rows)


def load_data(folder: str = "Data"):
    df_eff = pd.concat(
        [
            pd.read_csv(f"{folder}/databol{i}.csv", sep=";").assign(imputation=i)
            for i in range(1, 6)
        ]
    )
    df_eff["facine3"] = pd.to_numeric(df_eff["facine3"])
    df_eff["facine3"] /= 5
    replace_dict = {
        "bage": {
            1: "Under 35",
            2: "35-44",
            3: "45-54",
            4: "55-64",
            5: "65-74",
            6: "Over 75",
        },
        "percrent": {
            1: "< P20",
            2: "P20-P40",
            3: "P40-P60",
            4: "P60-80",
            5: "P80-P90",
            6: "> P90",
        },
        "nsitlabdom": {
            1: "Employee",
            2: "Self-Employed",
            3: "Retired",
            4: "Other Inactive or Unemployed",
        },
        "neducdom": {
            1: "Below Secondary Education",
            2: "Secondary Education",
            3: "University Education",
        },
        "np2_1": {1: "Ownership", 2: "Other"},
        # "nnumadtrab": {0: "None", 1: "One", 2: "Two", 3: "Three or More"},
        # PEOPLE_IN_HOUSEHOLD: {5: "5 or more"},  # 5 is 5 or more but we are treating it as 5
        # "percriq": {1: "< P25", 2: "P25-P50", 3: "P50-P75", 4: "P75-P90", 5: "> P90"},
    }
    # df_eff[PEOPLE_IN_HOUSEHOLD] = df_eff[PEOPLE_IN_HOUSEHOLD].astype(int)

    df_eff = df_eff.replace(to_replace=replace_dict)
    df_eff[PEOPLE_IN_HOUSEHOLD] = pd.to_numeric(df_eff[PEOPLE_IN_HOUSEHOLD])
    df_eff[Num_Workers] = pd.to_numeric(df_eff[Num_Workers])

    _log_missing(df_eff, ["facine3", PEOPLE_IN_HOUSEHOLD, Num_Workers])
    
    return df_eff


def _log_missing(df: pd.DataFrame, cols: list[str]) -> None:
    miss = df[cols].isna().sum()
    if miss.any():
        print("⚠️  Missing values:")
        print(miss[miss > 0])


if __name__ == "__main__":
    df_eff = load_data()
    check_nan(df_eff, Num_Workers)


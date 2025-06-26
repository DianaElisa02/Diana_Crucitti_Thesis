def top_share(df, col, weight, pct):
    df = df[[col, weight]].dropna().sort_values(col, ascending=False).copy()
    df["cum_weight"] = df[weight].cumsum()
    cutoff = df[weight].sum() * pct
    df["in_top"] = df["cum_weight"] <= cutoff
    top_sum = (df.loc[df["in_top"], col] * df.loc[df["in_top"], weight]).sum()
    total = (df[col] * df[weight]).sum()
    return top_sum / total

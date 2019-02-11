def remove_outliers(df, feature_cols, n):
    for c in feature_cols:
        df = df[df[c] <= df[c].mean() + n * df[c].std()]
    return df


import pandas as pd

def check_null(df):
    missing_vals = pd.DataFrame()
    missing_vals['Number of Nulls'] = df.isna().sum()
    missing_vals['% Null'] = (df.isna().sum() / len(df)) * 100
    return missing_vals
    

def check_unique(df, col, dropna=False):
    if dropna:
        unique_vals = pd.DataFrame(df[col].value_counts())
    else:
        unique_vals = pd.DataFrame(df[col].value_counts(dropna=False))
    return unique_vals
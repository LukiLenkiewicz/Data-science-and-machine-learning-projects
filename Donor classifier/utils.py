import numpy as np
import pandas as pd

def fill_empty(df, mean_=True):

    for column in df.columns:
        if df.dtypes[column] == "object":
            mode_val = df[column].mode()
            df[column].fillna(value=mode_val[0], inplace=True)
        else:
            mean_val = df[column].mean() if mean_ else df[column].median()
            df[column].fillna(value=mean_val, inplace=True)            

    return df


def load_data(filename):
    return pd.read_csv(filename, low_memory=False)


def calculate_profit(y_true, y_pred):
    profit = 0.
    for t, p in zip(y_true, y_pred):
        if t==1 and p ==1:
            profit += 13.
        if p == 1:
            profit -= 0.69
    return profit

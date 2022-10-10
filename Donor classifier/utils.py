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


def dataset_balancer(X_train, y_train, method='undersampler'):
    if method == 'undersampler':
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        return X_train, y_train
    elif method == 'oversampler':
        over_sampler = SMOTE(random_state=42)
        X_train, y_train = over_sampler.fit_resample(X_train, y_train)
    else:
        return X_train, y_train
    
    
def find_redundant_features(corr_matrix, corr_coef= .9):
    corr_matrix_columns = corr_matrix.columns    
    redundant_features = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            if corr_matrix.iloc[i, j] >= corr_coef:
                redundant_features.append(corr_matrix_columns[j])
    return redundant_features
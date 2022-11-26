import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


def load_data(filename):
    return pd.read_csv(filename, low_memory=False)


def fill_empty(df, mean_=True):

    for column in df.columns:
        if df.dtypes[column] == "object":
            mode_val = df[column].mode()
            df[column].fillna(value=mode_val[0], inplace=True)
        else:
            mean_val = df[column].mean() if mean_ else df[column].median()
            df[column].fillna(value=mean_val, inplace=True)            

    return df


def find_redundant_features(corr_matrix, corr_coef= .9):
    corr_matrix_columns = corr_matrix.columns    
    redundant_features = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            if corr_matrix.iloc[i, j] >= corr_coef:
                redundant_features.append(corr_matrix_columns[j])
    return redundant_features


def get_dependent_features(df, target_value, alpha=0.05):
    dependent_columns = []
    features = df.columns.to_list()
    features.remove(target_value)
    
    for column in features:
        _, p, _, _ = chi2_contingency(pd.crosstab(index=df[target_value], columns=df[column]).to_numpy())
        if p <= alpha:
            dependent_columns.append(column)
    
    return dependent_columns


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
    

def select_best_features(categorical_features, numerical_features, targets, no_of_categorical, no_of_numerical, num_method='test'):
    kb_cat = SelectKBest(chi2, k=no_of_categorical)
    X_cat = kb_cat.fit_transform(categorical_features, targets)
    if num_method == 'test':
        kb_num = SelectKBest(f_classif, k=no_of_numerical)
        X_num = kb_num.fit_transform(numerical_features, targets)
    elif num_method == 'corr':
        X_num = numerical_features.iloc[:, :no_of_numerical]
    X = np.concatenate((X_cat, X_num), axis=1)
    return X
    

def calculate_profit(y_true, y_pred):
    profit = 0.
    for t, p in zip(y_true, y_pred):
        if t==1 and p ==1:
            profit += 13.
        if p == 1:
            profit -= 0.69
    return profit

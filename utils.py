import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import List


def drop_outlier(ser: pd.Series) -> pd.Series:
    """
        Find and drop outlier 
    """
    q1, q3 = ser.quantile(q=[0.25, 0.75]).values
    iqr = q3 - q1
    upper_out = q3 + (1.5 * iqr)
    lower_out = q1 - (1.5 * iqr)
    return ser.loc[(ser >= lower_out) & (ser <= upper_out)]


def find_recency_last_visit(df: pd.DataFrame, col: str, today: str):
    """
        Find recency from last visit date
        ---
        today: YYYY-mm-dd format.
    """
    print(f"Find recency from last visit date {today}")
    today = datetime.strptime(today, '%Y-%m-%d')
    df[col] = df[col].map(pd.to_datetime)
    return df[col].map(lambda x: (today - x).days)


def create_transformer(t_type: str=['impute', 'standard', 'ohe'],
                      fill_value=None,
                      categories='auto' or List[str],
                      drop=None,
                      handle_unknown: str='error'):
    """
        Create transformer object
        ---
        fill_value: `impute` t_type
        categories: `ohe` t_type
        drop: `ohe` t_type {‘first’, ‘if_binary’}
        handle_unknown: `ohe` t_type {'error', 'ignore'}
    """
    if ((t_type == 'impute') and (fill_value != None)):
        return SimpleImputer(strategy='constant', fill_value=fill_value, verbose=1)
    elif t_type == 'standard':
        return StandardScaler()
    else:
        return OneHotEncoder(categories=categories, drop=drop, handle_unknown=handle_unknown)


def create_transformer_pipeline(name: List[str], transformer):
    """
        Create transformer pipeline from Sklearn
        ---
        transformer: List of transformer
    """
    assert len(name) == len(transformer) # same length
    steps = []
    for n, t in zip(name, transformer):
        steps.append((n, t))
    return Pipeline(steps, verbose=True)


def map_transformer(name: List[str], transformer_pipe, cols: List[List[str]],
                   remainder: str='drop'):
    """
        Map transfomer pipeline to columns
        ---
        transformer: List of transformer
        remainder: {‘drop’, ‘passthrough’}
    """
    assert len(name) == len(transformer_pipe) == len(cols) # same length
    transformers = []
    for n, t, c in zip(name, transformer_pipe, cols):
        transformers.append((n, t, c))
    return ColumnTransformer(transformers, remainder=remainder, n_jobs=-1, verbose=True)


def transformer_preprocess_template():
    """
        Template for creating transformer proprocessing
        ---
        Copy and modify to specific task and declare list of columns name before.
    """
    imp_r = create_transformer('impute', 365)
    imp_fm = create_transformer('impute', 0)
    scaler = create_transformer('standard')
    enc_flag = create_transformer('ohe', categories=np.repeat([[-1, 0, 1]], len(flag_cols), axis=0), drop='first')

    pipe_r = create_transformer_pipeline(['imp_r', 'scale_r'], [imp_r, scaler])
    pipe_fm = create_transformer_pipeline(['imp_fm', 'scale_fm'], [imp_fm, scaler])
    pipe_pref = create_transformer_pipeline(['scale_pref'], [scaler])
    pipe_flag = create_transformer_pipeline(['enc_flag'], [enc_flag]) 

    ct = map_transformer(['pipe_r', 'pipe_fm', 'pipe_pref', 'pipe_flag'],
                       [pipe_r, pipe_fm, pipe_pref, enc_flag],
                       [r_cols, fm_cols, pref_cols, flag_cols]) 

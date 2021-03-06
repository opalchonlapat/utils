import numpy as np
import pandas as pd
from datetime import datetime, date
from fuzzywuzzy import fuzz
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import List


def drop_outlier(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
        Drop NA first and Find and drop outlier 
    """
    df = df.dropna(subset=[col])
    q1, q3 = df[col].quantile(q=[0.25, 0.75]).values
    iqr = q3 - q1
    upper_out = q3 + (1.5 * iqr)
    lower_out = q1 - (1.5 * iqr)
    return df.loc[(df[col] >= lower_out) & (df[col] <= upper_out)]


def abbreviation_number(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


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


def fuzzy_dict(df: pd.DataFrame, col: str, target: list, fuzz_fn, th: int=90):
    """
        Find edit distance of real name
    """
    values = df[col].dropna().unique()
    result = {}
    for i in values:
        for j in target:
            if fuzz_fn(i,j) > th:
                result[i] = j
    return result


def clean_province(df: pd.DataFrame, province_col: str) -> pd.DataFrame:
    """
        Edit province name and convert to EN
    """
    province_df = pd.read_csv('https://github.com/PyThaiNLP/pythainlp/raw/dev/pythainlp/corpus/thailand_provinces_th.csv',
                               header=None, names=['name_th', 'abbr_th', 'name_en', 'abbr_en'])
    province_name = province_df[['name_th', 'name_en']].values.ravel()
    province_map_dict = fuzzy_dict(df, province_col, province_name, fuzz.partial_ratio, 90)
    province_th_en_map = province_df.set_index('name_th').to_dict()['name_en']
    df[province_col] = df[province_col].map(province_map_dict).map(province_th_en_map)
    return df


def find_recency(Y:int, m:int, d:int, df:pd.DataFrame, col_name:str) -> pd.Series:
    init_date = date(Y, m, d)
    recency = init_date - pd.to_datetime(df[col_name]).dt.date
    return recency.dt.days
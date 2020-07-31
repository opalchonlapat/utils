import pandas as pd


def drop_outlier(ser: pd.Series) -> pd.Series:
    """
        Find and drop outlier 
    """
    q1, q3 = ser.quantile(q=[0.25, 0.75]).values
    iqr = q3 - q1
    upper_out = q3 + (1.5 * iqr)
    lower_out = q1 - (1.5 * iqr)
    return ser.loc[(ser >= lower_out) & (ser <= upper_out)]
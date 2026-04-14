import pandas as pd
import numpy as np

def preprocess(df):
    df['log_amount'] = np.log1p(df['Amount'])
    df['hour'] = df['Time'] // 3600

    df = df.drop(['Time', 'Amount'], axis=1)

    return df
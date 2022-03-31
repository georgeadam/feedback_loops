import os
import pandas as pd

from settings import ROOT_DIR
from typing import Any, Dict


def load_support2cls_data(*args) -> Dict[str, Any]:
    df = pd.read_csv(os.path.join(ROOT_DIR, 'support2.csv'))
    one_hot_encode_cols = ['sex', 'dzclass', 'race', 'ca', 'income']
    target_variables = ['hospdead']
    remove_features = ['death', 'slos', 'd.time', 'dzgroup', 'charges', 'totcst',
                        'totmcst', 'aps', 'sps', 'surv2m', 'surv6m', 'prg2m',
                       'prg6m',
                       'dnr', 'dnrday', 'avtisst', 'sfdm2']
    df = df.drop(remove_features, axis=1)
    rest_colmns = [c for c in df.columns if c not in (one_hot_encode_cols + target_variables)]
    # Impute the missing values for 0.
    df[rest_colmns] = df[rest_colmns].fillna(0.)
    df = pd.get_dummies(df, prefix=one_hot_encode_cols)
    X_df = df.drop(target_variables, axis=1)
    X_df = X_df.astype('float64')

    y_df = df[target_variables[0]]
    dataset = {
        'problem': 'classification',
        'X': X_df,
        'y': y_df,
        'd_name': 'support2cls',
    }

    return dataset
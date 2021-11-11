import os
import pandas as pd

from settings import ROOT_DIR
from typing import Any, Dict


def load_mimic_iii_data(*args: Any, **kargs: Any) -> Dict[str, Any]:
    df_adult = pd.read_csv(os.path.join(ROOT_DIR, 'adult_icu.gz'), compression='gzip')

    train_cols = [
        'age', 'first_hosp_stay', 'first_icu_stay', 'eth_asian',
        'eth_black', 'eth_hispanic', 'eth_other', 'eth_white',
        'admType_ELECTIVE', 'admType_EMERGENCY', 'admType_NEWBORN',
        'admType_URGENT', 'heartrate_min', 'heartrate_max', 'heartrate_mean',
        'sysbp_min', 'sysbp_max', 'sysbp_mean', 'diasbp_min', 'diasbp_max',
        'diasbp_mean', 'meanbp_min', 'meanbp_max', 'meanbp_mean',
        'resprate_min', 'resprate_max', 'resprate_mean', 'tempc_min',
        'tempc_max', 'tempc_mean', 'spo2_min', 'spo2_max', 'spo2_mean',
        'glucose_min', 'glucose_max', 'glucose_mean', 'aniongap', 'albumin',
        'bicarbonate', 'bilirubin', 'creatinine', 'chloride', 'glucose',
        'hematocrit', 'hemoglobin', 'lactate', 'magnesium', 'phosphate',
        'platelet', 'potassium', 'ptt', 'inr', 'pt', 'sodium', 'bun', 'wbc']

    label = 'mort_icu'
    X_df = df_adult[train_cols]
    y_df = df_adult[label]

    dataset = {
        'problem': 'classification',
        'X': X_df,
        'y': y_df,
        'd_name': 'mimic_iii',
    }

    return dataset

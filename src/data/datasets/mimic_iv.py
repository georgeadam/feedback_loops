import numpy as np
import os
import pandas as pd

from settings import ROOT_DIR
from typing import Any, Dict


mimic_iv_paths = {"mimic_iv": {"path": "mimic_iv_datasets_with_year_imputed.csv", "categorical": False},
                  "mimic_iv_12h": {"path": "mimic_iv_datasets_with_year_12hrs_imputed.csv", "categorical": False},
                  "mimic_iv_24h": {"path": "mimic_iv_datasets_with_year_24hrs_imputed.csv", "categorical": False},
                  "mimic_iv_demographic": {"path": "mimic_iv_datasets_with_year_48hrs_imputed_age_race_gender.csv", "categorical": True},
                  "mimic_iv_12h_demographic": {"path": "mimic_iv_datasets_with_year_12hrs_imputed_age_race_gender.csv", "categorical": True},
                  "mimic_iv_24h_demographic": {"path": "mimic_iv_datasets_with_year_24hrs_imputed_age_race_gender.csv", "categorical": True}}



def load_mimic_iv_data(path: str, categorical: bool=False, model: str="lr") -> Dict[str, Any]:
    df_adult = pd.read_csv(os.path.join(ROOT_DIR, path))

    train_cols = [
        'year',
        'HeartRate_Min', 'HeartRate_Max', 'HeartRate_Mean', 'SysBP_Min',
        'SysBP_Max', 'SysBP_Mean', 'DiasBP_Min', 'DiasBP_Max', 'DiasBP_Mean',
        'MeanBP_Min', "MeanBP_Max", "MeanBP_Mean", "RespRate_Min", "RespRate_Max",
        "RespRate_Mean", "TempC_Min", "TempC_Max", "TempC_Mean", "SpO2_Min", "SpO2_Max",
        "SpO2_Mean", "Glucose_Min", "Glucose_Max", "Glucose_Mean", "ANIONGAP", "ALBUMIN",
        "BICARBONATE", "BILIRUBIN", "CREATININE", "CHLORIDE", "GLUCOSE", "HEMATOCRIT",
        "HEMOGLOBIN", "LACTATE", "MAGNESIUM", "PHOSPHATE", "PLATELET", "POTASSIUM", "PTT",
        "INR", "PT", "SODIUM", "BUN", "WBC"]
    if categorical:
        train_cols.append("age")

    categorical_cols = ["gender", "ethnicity"]

    if categorical:
        train_cols += categorical_cols

    label = 'mort_icu'
    X_df = df_adult[train_cols]

    if categorical and model != "random_forest":
        X_df = pd.get_dummies(X_df, columns=categorical_cols)
        X_df["age"] = X_df["age"].map({"0 - 10": 5, "10 - 20": 15, "20 - 30": 25, "30 - 40": 35,
                                       "40 - 50": 45, "50 - 60": 55, "60 - 70": 65, "70 - 80": 75, "> 80": 85})
    elif categorical:
        for col in categorical_cols:
            unique_vals = np.unique(X_df[col])
            replace = np.arange(len(unique_vals))
            mapping = {unique_vals[i]: replace[i] for i in range(len(unique_vals))}
            X_df[col] = X_df[col].map(mapping)

        X_df["age"] = X_df["age"].map({"0 - 10": 5, "10 - 20": 15, "20 - 30": 25, "30 - 40": 35,
                                       "40 - 50": 45, "50 - 60": 55, "60 - 70": 65, "70 - 80": 75, "> 80": 85})

    y_df = df_adult[label]

    dataset = {
        'problem': 'classification',
        'X': X_df,
        'y': y_df,
        'd_name': 'mimic_iv',
    }

    return dataset
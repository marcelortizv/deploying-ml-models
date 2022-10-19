"""
Check how is the performance of the model in slices of data

Date: Oct 2022
Author: Marcelo Ortiz
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
from src import functions
import logging

# paths
DATA_PATH = "data/clean/census.csv"
MODEL_PATH = "data/model/model.joblib"
ENCODER_PATH = "data/model/encoder.joblib"
LB_PATH = "data/model/lb.joblib"


def check_score_slices():
    """
    Execute score checking in slices
    """
    df = pd.read_csv(DATA_PATH)
    _, test = train_test_split(df, test_size=0.20)

    model = load(MODEL_PATH)
    encoder = load(ENCODER_PATH)
    lb = load(LB_PATH)

    slice_values = []

    for cat in functions.get_cat_features():
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = functions.process_data(
                df_temp,
                categorical_features=functions.get_cat_features(),
                label="salary", encoder=encoder, lb=lb, training=False)

            y_preds = model.predict(X_test)

            prc, rcl, fb = functions.compute_model_metrics(y_test, y_preds)

            line = "[%s->%s] Precision: %s " \
                   "Recall: %s FBeta: %s" % (cat, cls, prc, rcl, fb)
            logging.info(line)
            slice_values.append(line)

    with open('model/slice_performance_analysis.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')

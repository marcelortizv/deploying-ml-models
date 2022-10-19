"""
This script holds the training process of ML model

Date: Oct 2022
Author: Marcelo Ortiz
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from src import functions


def train_test_model():
    """
    Execute model training
    """
    df = pd.read_csv("data/clean/census.csv")
    train, test = train_test_split(df, test_size=0.20)

    X_train, y_train, encoder, lb = functions.process_data(
        train, categorical_features=functions.get_cat_features(),
        label="salary", training=True)

    model = functions.train_model(X_train, y_train)

    dump(model, "model/model.joblib")
    dump(encoder, "model/encoder.joblib")
    dump(lb, "model/lb.joblib")

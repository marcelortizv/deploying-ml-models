"""
This script holds unit test for functions
"""

import pandas as pd
import pytest
from src import functions
from joblib import load


@pytest.fixture
def data():
    """
    Get cleaned dataset
    """
    df = pd.read_csv("data/clean/census.csv")
    return df


def test_process_data(data):
    """
    Check if split have same number of rows for X and y
    """
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    X_test, y_test, _, _ = functions.process_data(
        data,
        categorical_features=functions.get_cat_features(),
        label="salary", encoder=encoder, lb=lb, training=False)

    assert len(X_test) == len(y_test)


def test_process_encoder(data):
    """
    Check if encoder get the same params
    """
    encoder_test = load("model/encoder.joblib")
    lb_test = load("model/lb.joblib")

    _, _, encoder, lb = functions.process_data(
        data,
        categorical_features=functions.get_cat_features(),
        label="salary", training=True)

    _, _, _, _ = functions.process_data(
        data,
        categorical_features=functions.get_cat_features(),
        label="salary", encoder=encoder_test, lb=lb_test, training=False)

    assert encoder.get_params() == encoder_test.get_params()
    assert lb.get_params() == lb_test.get_params()

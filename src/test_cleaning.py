"""
This script holds unit test for basic cleaning
"""

import pandas as pd
import pytest
import src.cleaning_data


@pytest.fixture
def data():
    """
    Fixtute to get raw dataset
    """
    df = pd.read_csv("data/census.csv", skipinitialspace=True)
    df = src.cleaning_data.clean_dataset(df)
    return df


def test_null(data):
    """
    This test if Data have no null values
    """
    assert data.shape == data.dropna().shape


def test_question_mark(data):
    """
    This test if Data has no question marks value
    """
    assert '?' not in data.values


def test_removed_columns(data):
    """
    This test if dropped column are no present in data
    """
    assert "fnlgt" not in data.columns
    assert "education-num" not in data.columns
    assert "capital-gain" not in data.columns
    assert "capital-loss" not in data.columns

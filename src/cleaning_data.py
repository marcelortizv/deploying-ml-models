"""
this script performs a cleaning data and create new data cleaned

Date: Oct 2022
Author: Marcelo Ortiz
"""

import pandas as pd


def clean_dataset(df):
    """
    Clean the dataset and drop unnecessary data
    """
    df.replace({'?': None}, inplace=True)
    df.dropna(inplace=True)
    df.drop("fnlgt", axis="columns", inplace=True)
    df.drop("education-num", axis="columns", inplace=True)
    df.drop("capital-gain", axis="columns", inplace=True)
    df.drop("capital-loss", axis="columns", inplace=True)
    return df


def execute_cleaning():
    """
    Execute data cleaning
    """
    df = pd.read_csv("data/census.csv", skipinitialspace=True)
    df = clean_dataset(df)
    df.to_csv("data/clean/census.csv", index=False)

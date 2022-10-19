"""
This script holds the used functions to run the ML training

Date: Oct 2022
Author: Marcelo Ortiz
"""

import logging
import numpy as np

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def get_cat_features():
    """
    Get categorical features
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
    return cat_features


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None,
    lb=None
):
    """ Process the data used in the machine learning pipeline.
    Processes the data using one hot encoding for the categorical features
    and a label binarizer for the labels. This can be used in either training or
    inference/validation.
    Args:

    :X : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`
    :categorical_features : list[str]
        List containing the names of the categorical features (default=[])
    :label : str
        Name of the label column in `X`. If None, then an empty array will
        be returned
        for y (default=None)
    :training : bool
        Indicator if training mode or inference/validation mode.
    :encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    :lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    Returns
    :data : np.array
        Processed data.
    :y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    :encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the
        encoder passed in.
    :lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the
        binarizer passed.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    categorical_data = X[categorical_features].values
    continuous_data = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        categorical_data = encoder.fit_transform(categorical_data)
        y = lb.fit_transform(y.values).ravel()
    else:
        categorical_data = encoder.transform(categorical_data)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    data = np.concatenate([continuous_data, categorical_data], axis=1)

    return data, y, encoder, lb


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Inputs

    Args:
    :X_train : np.array
        Training data.
    :y_train : np.array
        Labels.
    Returns:
    : model
        Trained ML model.
    """
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy',
                             cv=cv, n_jobs=-1)
    mean = np.mean(scores)
    std = np.std(scores)
    logging.info('Accuracy: %.3f (%.3f)' % (mean, std))
    return model


def compute_model_metrics(y, predictions):
    """
    Validates the trained machine learning model using precision, recall,
    and F1.
    Args
    :y : np.array
        Known labels, binarized.
    :predictions : np.array
        Predicted labels, binarized.
    Returns
    :precision : float
    :recall : float
    :fbeta : float
    """

    f_beta = fbeta_score(y, predictions, beta=1, zero_division=1)
    precision = precision_score(y, predictions, zero_division=1)
    recall = recall_score(y, predictions, zero_division=1)
    return precision, recall, f_beta


def inference(model, data):
    """ Run model inferences and return the predictions.
    Args
    :model :
        Trained machine learning model.
    :data : np.array
        Data used for prediction.
    Returns
    :y_preds : np.array
        Predictions from the model.
    """
    y_preds = model.predict(data)
    return y_preds
import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    precision = 0
    recall = 0
    f1 = 0
    accuracy = np.sum(y_pred == y_true) / y_pred.shape[0]
    if np.sum(y_pred == "1") != 0:
        precision = np.sum(np.logical_and(y_pred == y_true, y_pred == "1")) / np.sum(
            y_pred == "1"
        )
    if np.sum(y_true == "1") != 0:
        recall = np.sum(np.logical_and(y_pred == y_true, y_pred == "1")) / np.sum(
            y_true == "1"
        )
    if precision != 0 or recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    accuracy = np.sum(y_pred == y_true) / y_pred.shape[0]
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    ss_res = np.sum((y_true - y_pred)**2) 
    ss_tot = np.sum((y_true - np.mean(y_true))**2) 
    r2 = 1 - (ss_res / ss_tot) 
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    ss_res = np.sum((y_true - y_pred)**2) 
    return ss_res/y_true.shape[0]


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    ss_res = np.sum(np.abs(y_true - y_pred)) 
    return ss_res/y_true.shape[0]
    

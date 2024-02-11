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

    tp_value = np.count_nonzero((y_pred == '1') & (y_true == '1'))
    fn_value = np.count_nonzero((y_pred == '0') & (y_true == '1'))
    tn_value = np.count_nonzero((y_pred == '0') & (y_true == '0'))
    fp_value = np.count_nonzero((y_pred == '1') & (y_true == '0'))

    check_zero_p = tp_value + fp_value
    check_zero_r = tp_value + fn_value
    
    if (check_zero_p == 0) or (check_zero_r ==0):
        raise ZeroDivisionError('Check your data')
    
    precision = tp_value / (tp_value + fp_value)
    recall = tp_value / (tp_value + fn_value)
    f1 = (2 * tp_value) / (2 * tp_value + fp_value + fn_value)
    accuracy = (tp_value + tn_value) / (tp_value + tn_value + fp_value + fn_value)

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
    
    true_predictions = (y_pred == y_true).sum()
    
    return true_predictions / y_true.shape[0]


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    
    mean_value = np.mean(y_true)
    
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - mean_value)**2))


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    
    return np.mean((y_pred - y_true)**2)


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    return np.mean(np.abs(y_pred - y_true))
    
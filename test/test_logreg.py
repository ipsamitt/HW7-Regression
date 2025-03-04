"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from regression.logreg import LogisticRegressor 
from regression.utils import loadDataset # Assuming your implementation is in logistic_regressor.py

    # Load data
X_train, X_val, y_train, y_val = loadDataset(
    features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
    ],
    split_percent=0.8,
    split_seed=42
)

    # Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

log_model = LogisticRegressor(num_feats=X_train.shape[1], learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
log_model.train_model(X_train, y_train, X_val, y_val)

def test_prediction():
    """ Check that predictions are between 0 and 1. """
    y_pred = log_model.make_prediction(X_val)
    print(y_pred)
    assert np.all((y_pred >= 0) & (y_pred <= 1)), "Predictions should be in [0,1] range"


def test_loss_function():
    """ Check that loss function is computed correctly with a known example. """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2])
    expected_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    computed_loss = log_model.loss_function(y_true, y_pred)
    assert np.isclose(computed_loss, expected_loss), f"Loss mismatch: expected {expected_loss}, got {computed_loss}"


def test_gradient():
    """ Check that the gradient calculation works with a simple example. """
    X_test = np.array([[1, 2, 1], [4, 5, 1], [7, 8, 1]])  # Bias term included
    y_test = np.array([0, 1, 1])
    
    sc.W = np.zeros(X_test.shape[1])  # Set weights to zero
    grad = log_model.calculate_gradient(y_test, X_test)

    expected_grad = (X_test.T @ (log_model.make_prediction(X_test) - y_test)) / X_test.shape[0]
    
    assert np.allclose(grad, expected_grad), f"Gradient mismatch: expected {expected_grad}, got {grad}"


def test_training():
    """ Check that weights update after training starts. """
    initial_weights = log_model.W.copy()
    
    log_model.train_model(X_train, y_train, X_val, y_val)
    
    assert not np.array_equal(initial_weights, log_model.W), "Weights should update during training"

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

X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])  # Add bias term
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])        # Add bias term

log_model = LogisticRegressor(num_feats=X_train.shape[1], learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
log_model.train_model(X_train, y_train, X_val, y_val)

def test_prediction():
    """ Check that predictions are between 0 and 1. """
    y_pred = log_model.make_prediction(X_val)
    assert np.all((y_pred >= 0) & (y_pred <= 1)), "Predictions should be in [0,1] range"


def test_loss_function():
    """ Check that loss function is computed correctly with a known example. """
    y_true = y_val
    y_pred = log_model.make_prediction(X_val)
    expected_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    computed_loss = log_model.loss_function(y_true, y_pred)
    assert np.isclose(computed_loss, expected_loss), f"Loss mismatch: expected {expected_loss}, got {computed_loss}"


def test_gradient():
    """ Test if the computed gradient is close to the numerical gradient. """
    
    # Create a small test case
    num_feats = 3  # Number of features
    log_model = LogisticRegressor(num_feats)  # Initialize model
    
    # Generate a small synthetic dataset
    np.random.seed(42)
    X_test = np.random.randn(5, num_feats)  # 5 samples, 3 features
    y_test = np.random.randint(0, 2, 5)  # Binary labels (0 or 1)
    
    # Add bias column to X_test (since train_model normally does this)
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])  # Manually add bias
    
    # Compute the analytical gradient
    computed_grad = log_model.calculate_gradient(y_test, X_test)
    
    # Compute numerical gradient using finite differences
    epsilon = 1e-5
    numerical_grad = np.zeros_like(log_model.W)

    for i in range(len(log_model.W)):
        W_plus = log_model.W.copy()
        W_minus = log_model.W.copy()
        W_plus[i] += epsilon
        W_minus[i] -= epsilon

        log_model.W = W_plus
        loss_plus = log_model.loss_function(y_test, log_model.make_prediction(X_test))

        log_model.W = W_minus
        loss_minus = log_model.loss_function(y_test, log_model.make_prediction(X_test))

        numerical_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

    # Restore original weights
    log_model.W = computed_grad

    # Check if computed and numerical gradients are close
    assert np.allclose(computed_grad, numerical_grad, atol=1e-4), \
        f"Gradient check failed! Computed: {computed_grad}, Numerical: {numerical_grad}"


def test_training():
    """ Check that weights update after training starts. """
    initial_weights = log_model.W.copy()
    
    log_model.train_model(X_train, y_train, X_val, y_val)
    
    assert not np.array_equal(initial_weights, log_model.W), "Weights should update during training"

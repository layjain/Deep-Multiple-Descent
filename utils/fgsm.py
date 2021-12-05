import numpy as np

def fgsm_adversarial_data(X, dt, epsilon):
    return X+epsilon*np.sign(dt(X))

def loss_derivative_mse(X,Y,theta):
    """
    Calculate dJ/dX where J is mean squared error
    X : n x d
    Y : n x 1
    theta : d x 1
    """
    return 2*theta.T*(X*theta.T-Y)
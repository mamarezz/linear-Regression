import pandas as pd
import numpy as np
import matplotlib as plt 
data = pd.read_csv("E:\cods\liner regression\swedish_insurance.csv")
A = data['X'].values
Y = data['Y'].values
X = np.c_[np.ones(len(A)), A]

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape
    theta = np.zeros((n, 1))
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y.reshape(-1, 1)
        updates = X.T @ errors / m
        theta -= alpha * updates
#         print(theta)
    return np.round(theta.flatten(), 4)

def linear_regression_gradient_descentinput(X, Y, alpha=0.00003, iterations=1000):
    theta = linear_regression_gradient_descent(X, Y, alpha, iterations)
    print(f"Optimized theta: {theta}")
    return theta


theta = linear_regression_gradient_descentinput(X, Y)

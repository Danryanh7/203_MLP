import numpy as np

def sigmoid(z: np.ndarray):
    """
    Sigmoid functions 
    """
    return 1 / (1 + np.exp(-z))

def dSigmoid(z):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(z)*(1-sigmoid(z))
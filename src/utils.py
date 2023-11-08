import numpy as np

def sigmoid(z: np.ndarray):
    """
    Sigmoid functions 
    """
    return 1 / (1 + np.exp(-z))
import numpy as np

def sigmoid(x):
    x[np.where(x < -60)] = -60
    x[np.where(x > 60)] = 60
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)
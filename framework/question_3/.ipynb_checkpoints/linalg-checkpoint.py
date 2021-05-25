import numpy as np

from framework.question_3.module import Module

class FrobeniusNorm(Module):

    def __init__(self, linear):
        super().__init__()
        
        self.linear = linear
        self.g_weight = None

    def init_weights(self, seed=None):
        pass

    def forward(self, *xs):
        super().forward(None)

        weight_matrix = self.linear.weight

        return np.trace(weight_matrix.T @ weight_matrix)

    def backward(self, g_next_layer):
        weight_matrix = self.linear.weight

        self.g_weight = 2 * np.dot(g_next_layer, weight_matrix)
        return 0

    def update(self, learning_rate):
        weight_matrix = self.linear.weight

        weight_matrix -= learning_rate * self.g_weight
        
    def __call__(self, *xs):
        return self.forward(*xs)
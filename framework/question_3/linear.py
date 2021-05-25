import numpy as np

from framework.question_3.module import Module

class Linear(Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim  = in_dim
        self.out_dim = out_dim

        self.weight = np.zeros((in_dim, out_dim))
        self.bias   = np.zeros((out_dim, 1))

        self.g_weight = np.zeros(self.weight.shape)
        self.g_bias   = np.zeros(self.bias.shape)

    def init_weights(self, seed=None):
        np.random.seed(seed)
        
        self.weight = np.random.randn(*self.weight.shape)
        self.bias   = np.random.randn(*self.bias.shape)

    def forward(self, x):
        super().forward(x)
#         print(np.dot(x, self.weight) + self.bias.T)
        return np.dot(x, self.weight) + self.bias.T

    def backward(self, g_next_layer):

        self.g_weight = np.dot(self.input.T, g_next_layer)
        self.g_bias   = np.sum(g_next_layer, axis=0, keepdims=True)

        return np.dot(g_next_layer, self.weight.T)

    def update(self, learning_rate):

        self.weight -= learning_rate * self.g_weight
        self.bias   -= learning_rate * self.g_bias.T
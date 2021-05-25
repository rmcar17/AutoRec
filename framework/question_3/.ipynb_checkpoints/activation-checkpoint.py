import numpy as np

from framework.question_3.module import Module
from framework.question_3.functional import sigmoid, relu


class Sigmoid(Module):

    def __init__(self):
        super().__init__()

        self.sig_input = None

    def init_weights(self, seed=None):
        pass

    def forward(self, x):
        super().forward(x)

        self.sig_input = sigmoid(x)

        return self.sig_input

    def backward(self, g_next_layer):
        return self.sig_input * (1 - self.sig_input) * g_next_layer

    def update(self, learning_rate):
        pass

class ReLU(Module):

    def __init__(self):
        super().__init__()

    def init_weights(self, seed=None):
        pass

    def forward(self, x):
        super().forward(x)

        return relu(x)

    def backward(self, g_next_layer):
        return np.array(self.input > 0, dtype=np.float32) * g_next_layer

    def update(self, learning_rate):
        pass
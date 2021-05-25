import numpy as np

from framework.question_3.module import Module

class Sequential(Module):

    def __init__(self, modules):
        super().__init__()

        self.sequential = modules

    def init_weights(self, seed=None):
        
        for i, l in enumerate(self.sequential):
            if seed is None:
                l.init_weights(seed=None)
            else:
                l.init_weights(seed=seed+i)

    def forward(self, x):
        super().forward(x)

        f_val = x
        for l in self.sequential:
            f_val = l.forward(f_val)

        return f_val

    def backward(self, g_next_layer):
        b_grad = g_next_layer
        for l in reversed(self.sequential):
            b_grad = l.backward(b_grad)

        return b_grad

    def update(self, learning_rate):
        for l in self.sequential:
            l.update(learning_rate)
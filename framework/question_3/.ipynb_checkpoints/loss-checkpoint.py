import numpy as np

from abc import ABC

from framework.question_3.module import Module

class Loss(Module):
    def __init__(self):
        super().__init__()

    def init_weights(self, seed=None):
        pass

    def update(self, learning_rate):
        pass

    def __call__(self, *xs):
        return self.forward(*xs)

class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, *xs):
        super().forward(xs)

        predicted, target = xs

        return np.mean((predicted - target)**2)

    def backward(self, g_next_layer=1):
        predicted, target = self.input

        return -2 * (target - predicted) / target.shape[0] * g_next_layer
    
class WeightedSumLoss(Loss):
    def __init__(self, loss_weight_tuples):
        super().__init__()
        self.losses  = [l for l, _ in loss_weight_tuples]
        self.weights = [w for _, w in loss_weight_tuples]

    def forward(self, *xs):
        super().forward(xs)
        
        weighted_loss = sum(w * l(*xs) for (l, w) in zip(self.losses, self.weights))

        return weighted_loss

    def backward(self, g_next_layer=1):
        weighted_back_grad = sum(w * l.backward(g_next_layer) for (l, w) in zip(self.losses, self.weights))
        weighted_back_grad *= g_next_layer

        return weighted_back_grad

    def update(self, learning_rate):
        for l, w in zip(self.losses, self.weights):
            l.update(learning_rate * w)
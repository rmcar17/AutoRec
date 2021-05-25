from abc import ABC, abstractmethod

class Module(ABC):

    @abstractmethod
    def __init__(self):
        self.__input = None

    @property
    def input(self):
        return self.__input

    @input.setter
    def input(self, value):
        self.__input = value

    @abstractmethod
    def init_weights(self, seed=None):
        pass
    
    @abstractmethod
    def forward(self, x):
        self.input = x

    @abstractmethod
    def backward(self, g_next_layer):
        pass

    @abstractmethod
    def update(self, learning_rate):
        pass

    def __call__(self, x):
        return self.forward(x)
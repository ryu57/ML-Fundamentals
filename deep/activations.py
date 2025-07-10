import numpy as np

def select_activation(activation="relu"):
    if activation == "relu":
        return ReLU()
    elif activation == "sig":
        return Sigmoid()
    else:
        return ReLU()

class ReLU:
    def activate(self, x):
        mask = (x >= 0).astype(int)
        return mask * x
    
    def deriv(self, x):
        return (x >= 0).astype(int)

class Sigmoid:

    def activate(self, x):
        return 1 / (1 + np.exp(-x))
    
    def deriv(self, x):
        return self.activate(x) * (1 - self.activate(x))
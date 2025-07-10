import numpy as np

def select_activation(activation="relu"):
    if activation == "relu":
        return ReLU()
    elif activation == "sig":
        return Sigmoid()
    elif activation == "none":
        return NoActivation()
    else:
        return NoActivation()

class ReLU:
    def activate(self, X):
        mask = (X >= 0).astype(int)
        return mask * X
    
    def deriv(self, X):
        return (X >= 0).astype(int)

class Sigmoid:

    def activate(self, X):
        return 1 / (1 + np.exp(-X))
    
    def deriv(self, X):
        return self.activate(X) * (1 - self.activate(X))

class NoActivation:
    def activate(self, X):
        return X

    def deriv(self, X):
        return np.ones_like(X)
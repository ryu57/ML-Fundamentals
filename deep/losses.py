import numpy

def select_loss(loss="mse"):
    if loss == "mse":
        return MSE()
    else:
        return MSE()

class MSE:
    def loss(self, prediction, truth):
        return ((truth - prediction) ** 2).sum() / len(truth)

    def deriv(self, prediction, truth):
        return -2 * (truth - prediction) / len(prediction)
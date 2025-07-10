import numpy as np 
import activations
import losses

class FeedForward:
    def __init__(self, input_size, output_size, weight_init="he", activation="relu"):
        # Initialize weights
        if weight_init == "he": # He Initialization - ReLU
            std_dev = np.sqrt(2 / input_size)
            self.W = np.random.randn(output_size, input_size) * std_dev
        elif weight_init == "xa": # Xavier Initialization - sigmoid/tanh
            std_dev = np.sqrt(1 / (input_size + output_size))
            self.W = np.random.randn(output_size, input_size) * std_dev
        else: # Default He Initialization
            std_dev = np.sqrt(2 / input_size)
            self.W = np.random.randn(output_size, input_size) * std_dev

        self.b = np.zeros(output_size)

        self.activation = activations.select_activation(activation)
        
        self.z = None
        self.a = None
        self.x = None


    def forward(self, x):
        self.x = x
        self.z = np.dot(self.W, x) + self.b
        self.a = self.activation.activate(self.z)
        return self.a

    def backward(self, truth, lr=0.01, loss = "mle"):
        loss_fn = losses.select_loss(loss)
        dLda = loss_fn.deriv(self.a, truth)
        dadz = self.activation.deriv(self.z)
        dLdW = np.outer(dLda * dadz , self.x)
        dLdb = dLda * dadz * 1

        self.W = self.W - lr * dLdW
        self.b = self.b - lr * dLdb




if __name__ == "__main__":
    layer = FeedForward(input_size=4, output_size=3, weight_init="he", activation="sig")
    truth = np.array([1,1,1])
    input_data = np.array([1,2,3,4])
    lr = 0.1
    loss = "mse"

    for i in range(100):
        result = layer.forward(input_data)
        print(result)
        layer.backward(truth,lr=lr, loss=loss)






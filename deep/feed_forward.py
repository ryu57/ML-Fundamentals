import numpy as np 
import activations
import losses

class DenseLayer:
    def __init__(self, input_size, output_size, use_bias=True, weight_init="he", activation="relu"):
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
        self.use_bias = use_bias

        self.activation = activations.select_activation(activation)
        
        self.Z = None
        self.A = None
        self.X = None


    def forward(self, X):
        self.X = X
        if self.use_bias:
            self.Z = np.dot(X, self.W.T) + self.b
        else:
            self.Z = np.dot(X, self.W.T)

        self.A = self.activation.activate(self.Z)
        return self.A
    
    def backward(self, dL_dA, lr=0.01):
        # dL_dA: upstream gradient (shape: batch_size x output_size)

        # Derivative of activation
        dAdZ = self.activation.deriv(self.Z)  # same shape as output

        # Gradient of loss w.r.t pre-activation output
        dL_dZ = dL_dA * dAdZ  # element-wise

        # Gradients w.r.t weights and biases
        dL_dW = np.dot(dL_dZ.T, self.X) / self.X.shape[0]
        dL_db = np.sum(dL_dZ, axis=0) / self.X.shape[0]

        # Update weights
        self.W -= lr * dL_dW
        self.b -= lr * dL_db

        # Gradient w.r.t input to propagate backward
        dL_dX = np.dot(dL_dZ, self.W)  # shape: batch_size x input_size

        return dL_dX



if __name__ == "__main__":
    layer = DenseLayer(input_size=4, output_size=3, weight_init="he", activation="sig")
    truth = np.array([[1,1,1]])
    input_data = np.array([[1,2,3,4]])
    lr = 0.1
    loss_fn = losses.select_loss(loss="mse")

    for i in range(100):
        output = layer.forward(input_data)
        print(output)

        loss = loss_fn.loss(output, truth)
        dL_dA = loss_fn.deriv(output, truth)

        layer.backward(dL_dA,lr=lr)






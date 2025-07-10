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
        
        self.Z = None
        self.A = None
        self.X = None


    def forward(self, X):
        self.X = X
        self.Z = np.dot(X, self.W.T) + self.b
        self.A = self.activation.activate(self.Z)
        return self.A
    
    def backward(self, truth, lr=0.01, loss="mse"):
        loss_fn = losses.select_loss(loss)
        
        # Gradients w.r.t. output
        dLdA = loss_fn.deriv(self.A, truth)  # Shape: (n, d_out)
        dAdZ = self.activation.deriv(self.Z)  # Shape: (n, d_out)
        dLdZ = dLdA * dAdZ  # Shape: (n, d_out)
        
        # Gradients w.r.t. weights and biases
        dLdW = np.dot(dLdZ.T, self.X) / self.X.shape[0]  # Averaged over batch
        dLdb = np.sum(dLdZ, axis=0) / self.X.shape[0]    # Sum over rows, then average
        
        print("Z before")
        print(self.Z)

        # Update parameters
        self.W -= lr * dLdW
        self.b -= lr * dLdb

        print("Z after")
        print(np.dot(self.X, self.W.T) + self.b)




if __name__ == "__main__":
    layer = FeedForward(input_size=4, output_size=3, weight_init="he", activation="relu")
    truth = np.array([[1,1,1]])
    input_data = np.array([[1,2,3,4]])
    lr = 0.1
    loss = "mse"

    for i in range(3):
        result = layer.forward(input_data)
        print(result)
        layer.backward(truth,lr=lr, loss=loss)






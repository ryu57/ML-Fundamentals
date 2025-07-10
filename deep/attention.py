import numpy as np 
from feed_forward import DenseLayer
import losses

class AttentionLayer:
    def __init__(self, input_size, dk=1):
        self.dk = dk

        self.linear_prop_Q = DenseLayer(input_size=input_size, output_size=dk, use_bias=False, weight_init="he", activation="none")
        self.linear_prop_K = DenseLayer(input_size=input_size, output_size=dk, use_bias=False, weight_init="he", activation="none")
        self.linear_prop_V = DenseLayer(input_size=input_size, output_size=dk, use_bias=False, weight_init="he", activation="none")

    def forward(self,X):
        self.Q = self.linear_prop_Q.forward(X)
        self.K = self.linear_prop_K.forward(X)
        self.V = self.linear_prop_V.forward(X)

        self.S = np.dot(self.Q, self.K.T) 
        self.S_scaled = self.S / self.dk ** 0.5

        self.Att = self._softmax(self.S_scaled)
        self.Z = np.dot(self.Att, self.V)
    
        return self.Z

    def backward(self, dL_dZ, lr=0.01):
        # dL_dZ: upstream gradient (T, dk)
        T = self.Q.shape[0]

        # Step 1: Gradients w.r.t. Attention weights and V
        dL_dAtt = np.dot(dL_dZ, self.V.T)  # (T, T)
        dL_dV = np.dot(self.Att.T, dL_dZ)  # (T, dk)

        # Step 2: Gradient through softmax
        # Using vectorized Jacobian trick for softmax gradient:
        # dL/dS = A * (dL/dA - sum(dL/dA * A))
        sum_dL_dAtt_times_Att = np.sum(dL_dAtt * self.Att, axis=1, keepdims=True)  # (T,1)
        dL_dS_scaled = self.Att * (dL_dAtt - sum_dL_dAtt_times_Att)  # (T,T)

        # Step 3: Gradient w.r.t. unscaled scores S
        dL_dS = dL_dS_scaled / np.sqrt(self.dk)  # scale gradient back

        # Step 4: Gradients w.r.t Q and K
        dL_dQ = np.dot(dL_dS, self.K)  # (T, dk)
        dL_dK = np.dot(dL_dS.T, self.Q)  # (T, dk)

        # Step 5: Backprop through linear projections (DenseLayers)
        # Each dense layer should implement its own backward which updates weights internally
        self.linear_prop_Q.backward(dL_dQ, lr=lr)
        self.linear_prop_K.backward(dL_dK, lr=lr)
        self.linear_prop_V.backward(dL_dV, lr=lr)

        # Step 6: Gradient w.r.t input X (optional, if used in earlier layers)
        dL_dX = (np.dot(dL_dQ, self.linear_prop_Q.W) +
                 np.dot(dL_dK, self.linear_prop_K.W) +
                 np.dot(dL_dV, self.linear_prop_V.W))  # (T, input_size)

        return dL_dX

    def _softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

if __name__ == "__main__":
    # Suppose your input_size (embedding dimension) = 3
    input_size = 3
    dk = 2  # output projection dimension
    truth = np.array([[1,  1],
            [1, 1],
            [1, 1],
            [1, 1]])
    loss_fn = losses.select_loss("mse")

    # Create a simple input matrix X with T=4 tokens, embedding size = 3
    X = np.array([
        [1.0, 0.0, 0.0],  # token 1 embedding
        [0.0, 1.0, 0.0],  # token 2 embedding
        [0.0, 0.0, 1.0],  # token 3 embedding
        [1.0, 1.0, 1.0],  # token 4 embedding
    ])

    # Initialize your attention layer
    attention = AttentionLayer(input_size=input_size, dk=dk)

    # Run forward pass
    for i in range(100):
        output = attention.forward(X)

        print("Output matrix:\n", output)

        dL_dZ = loss_fn.deriv(output, truth)

        attention.backward(dL_dZ, lr=0.1)





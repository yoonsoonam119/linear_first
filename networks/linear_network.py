import numpy as np

# This class defines a simple linear neural network with one hidden layer.
# It includes methods for forward propagation, backward propagation (gradient descent), 
# and training the network. The class can be initialized with custom weights or 
# with random weights if none are provided.

class LinearNetwork:
    def __init__(self, in_dim, hidden_dim, out_dim, init_w1 = None, init_w2 = None):

        if init_w1 is not None and init_w2 is not None:
            self.W1 = init_w1.copy()
            self.W2 = init_w2.copy()
        else:
            self.W1 = np.random.randn(hidden_dim, in_dim)
            self.W2 = np.random.randn(out_dim, hidden_dim)

    def forward(self, x): 
        self.z = self.W2 @ self.W1 @ x
        return self.z

    def backward(self, x, y, learning_rate):

        forward = self.W2 @ self.W1 @ x
        dW1 = 1/x.shape[1] * self.W2.T @ (forward-y) @ x.T 
        dW2 = 1/x.shape[1] * (forward - y) @ x.T @ self.W1.T 

        self.W2 -= learning_rate * dW2
        self.W1 -= learning_rate * dW1

    def train(self, X_train, Y_train, epochs, learning_rate):
        w1s = []
        w2s = []
        losses = []
        for _ in range(epochs):
            loss = np.mean((self.forward(X_train) - Y_train) ** 2)
            losses.append(loss)
            w1s.append(self.W1.copy())
            w2s.append(self.W2.copy())
            self.backward(X_train, Y_train, learning_rate)
        return w1s, w2s, losses
    
    def train_fast(self, X_train, Y_train, epochs, learning_rate):
        losses = []
        for _ in range(epochs):
            # loss = np.mean((self.forward(X_train) - Y_train) ** 2)
            
            loss = 1 / (2 * X_train.shape[1]) * np.linalg.norm(self.W2 @ self.W1 @ X_train - Y_train, ord='fro')**2 
            losses.append(loss)
            self.backward(X_train, Y_train, learning_rate)
        return losses


    



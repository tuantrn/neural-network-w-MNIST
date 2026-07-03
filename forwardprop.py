import numpy as np

# activation functions
def ReLU(x):
    return np.maximum(x, 0)

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def forward_propagation(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    
    return Z1, A1, Z2, A2


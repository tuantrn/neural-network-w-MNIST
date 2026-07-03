import numpy as np

# one-hot encode of Y
def one_hot(x):
    res = np.zeros((x.size, x.max() + 1))
    res[np.arange(x.size), x] = 1

    return res.T

def derivative_ReLU(x):
    return x > 0

def back_propagation(w2, Z1, A1, A2, X, Y):
    one_hot_Y = one_hot(Y)
    m = Y.size
    dZ2 = A2 - one_hot_Y
    dw2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m  * np.sum(dZ2)
    dZ1 = w2.T.dot(dZ2) * derivative_ReLU(Z1)
    dw1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dw1, db1, dw2, db2

def update_params(w1, dw1, b1, db1, w2, dw2, b2, db2, a):
    w1 = w1 - a * dw1
    b1 = b1 - a * db1
    w2 = w2 - a * dw2 
    b2 = b2 - a * db2
    
    return w1, b1, w2, b2


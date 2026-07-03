import numpy as np
from forwardprop import forward_propagation
from backprop import back_propagation, update_params

def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return w1, b1, w2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)

    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, a):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = back_propagation(w2, Z1, A1, A2, X, Y)
        w1, b1, w2, b2 = update_params(w1, dw1, b1, db1, w2, dw2, b2, db2, a)
        if (i % 10 == 0):
            print("Iterations: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))

    return w1, b1, w2, b2

def make_predictions(X, w1, b1, w2, b2):
    Z1, A1, Z2, A2 = forward_propagation(w1, b1, w2, b2, X)
    predictions = get_predictions(A2)
    
    return predictions


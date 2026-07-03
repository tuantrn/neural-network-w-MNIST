import zipfile
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nn import gradient_descent, make_predictions

# extract and shuffle dataset
target = "./mnist.zip"
with zipfile.ZipFile(target, 'r') as zip_ref:
    with zip_ref.open("mnist_train.csv") as mnist_train:
        data_train = pd.read_csv(mnist_train)
        data_train = np.array(data_train)
        np.random.shuffle(data_train)

    with zip_ref.open("mnist_test.csv") as mnist_test:
        data_dev = pd.read_csv(mnist_test)
        data_dev = np.array(data_dev)
        np.random.shuffle(data_dev)

# training set
data_train = data_train.T
Y_train = data_train[0]
X_train = data_train[1:785]
X_train = X_train / 255.

# validation set
data_dev = data_dev.T
Y_dev = data_dev[0]
X_dev = data_dev[1:785]
X_dex = X_dev / 255.

def test_predictions(index, w1, b1, w2, b2):
    current_image = X_dev[:, index, None]
    predictions = make_predictions((X_dev[:, index, None]), w1, b1, w2, b2)
    label = Y_dev[index]
    print("Predictions: ", predictions)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)

for i in range(10):
    test_predictions(i, w1, b1, w2, b2)

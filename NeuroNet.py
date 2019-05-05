import time

import cv2
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_files
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw  # Подключим необходимые библиотеки.


def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y_vect)):
        y_vect[i, y[i]] = 1
    return y_vect


def f(x):
    return 1 / (1 + np.exp(-x))


def df(x):
    return f(x) * (1 - f(x))


def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        test = (nn_structure[l], nn_structure[l - 1])
        W[l] = rand.random_sample((nn_structure[l], nn_structure[l - 1]))
        test = (nn_structure[l],)
        b[l] = rand.random_sample((nn_structure[l],))
    return W, b


def init_delta_values(nn_structure):
    delta_W = {}
    delta_b = {}
    for l in range(1, len(nn_structure)):
        delta_W[l] = np.zeros((nn_structure[l], nn_structure[l - 1]))
        delta_b[l] = np.zeros((nn_structure[l],))
    return delta_W, delta_b


def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise,
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l + 1] = W[l].dot(node_in) + b[l]  # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l + 1] = f(z[l + 1])  # h^(l) = f(z^(l))
    return h, z


def calc_out_layer_delta(y, h_out, z_out):
    return -(y - h_out) * df(z_out)


def calc_hidden_delta(delta_next, w_l, z_l):
    return np.dot(np.transpose(w_l), delta_next) * df(z_l)


def train_nn(nn_structure, X, y, timeStart, iter_num=3000, alpha=5):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt % 10 == 0:
            print('Iteration {} of {}, time: {}'.format(cnt, iter_num, time.time() - timeStart))
        tri_W, tri_b = init_delta_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, to be used in the
            # gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calc_out_layer_delta(y[i, :], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i, :] - h[l]))
                else:
                    if l > 1:
                        delta[l] = calc_hidden_delta(delta[l + 1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l + 1][:, np.newaxis], np.transpose(h[l][:, np.newaxis]))
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l + 1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0 / m * tri_W[l])
            b[l] += -alpha * (1.0 / m * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0 / m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y


def load_img(path):
    image = cv2.imread(path)
    result = np.zeros((len(image) * len(image[0])))
    for i in range(len(image)):
        for j in range(len(image[i])):
            result[i * len(image[i]) + j] = np.mean(image[i][j]) / 255
    return result


def scale(data, size=1000, max=15):
    for i in range(size):
        if size >= len(data):
            return data
        for j in range(len(data[i])):
            data[i][j] = data[i][j] / max

    return data

if __name__ == '__main__':
    # load and scale data
    digits = load_digits()
    X_scale = StandardScaler()
    # data = np.zeros((8, 64))
    # y = np.zeros(8)
    # for i in range(1, 9):
    #     data[i - 1] = load_img('test' + str(i) + '.png')
    #     y[i - 1] = 6
    X = X_scale.fit_transform(digits.data)

    #X = scale(digits.data, len(digits.data) - 1)

    # image = load_img('test.png')
    # data = digits.data
    # data[0] = image
    # res = X_scale.fit_transform(data)
    # image = res[0]
    # image = np.zeros((len(res) * len(res[0])))
    # for i in range(len(res)):
    #     for j in range(len(res[i])):
    #         test = res[i][j]
    #         image[i * len(res[i]) + j] = res[i][j]
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    # convert digits to vector
    y_vect_train = convert_y_to_vect(y_train)
    y_vect_test = convert_y_to_vect(y_test)
    # setup NN structure
    nn_structure = [64, 30, 10]
    # train

    W, b, avg_cost_fun = train_nn(nn_structure, X_train, y_vect_train, time.time())
    # make plot
    plt.plot(avg_cost_fun)
    plt.ylabel('Average J')
    plt.xlabel('Iteration number')
    plt.show()
    # get the prediction accuracy and print
    y_pred = predict_y(W, b, X_test, 3)
    print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))

    while True:
        image = load_img('test1.png')
        imageId = np.round(1000 * rand.random_sample())
        plt.gray()
        plt.matshow(digits.images[int(imageId)])
        plt.show()
        h, z = feed_forward(X[int(imageId)], W, b)
        try:
            print(np.argmax(h[len(nn_structure)]))
        except:
            pass

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def dsoftmax(x):
    return softmax(x) * (1 - softmax(x))
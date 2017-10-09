from functions import *
import numpy as np
import matplotlib.pyplot as plt

neurons = [784, 128, 10]
n_dim = len(neurons)
epochs = 100
eta = 0.1
batch_size = 15


def forward_propagate(x, w, b):
    outputs = [np.zeros(s.shape) for s in b]
    vals = [np.zeros(s.shape) for s in b]
    vals[0] = x
    for i in range(1, n_dim):
        # w = np.array(w)
        print(w)
        outputs[i] = w[i].dot(vals[i-1]) + b[i]
        # print(outputs)
        vals[i] = sigmoid(outputs[i])
        # print(vals)
    return outputs, vals


def back_propagate(outputs, vals, x, y, w, b):
    dw = [np.zeros(s.shape) for s in w]
    db = [np.zeros(s.shape) for s in b]
    err = dsigmoid(outputs[-1]) * (vals[-1] - y)
    db[-1] = err
    dw[-1] = err.dot(vals[-2].transpose())

    for i in range(n_dim - 2, 0, -1):
        # print(self.w)
        # print(len(w))
        # print(len(w[0]))
        print(w[i+1])
        print(err)
        print(len(w[i+1]))
        print(len(err))
        err = np.multiply(w[i + 1].transpose().dot(err), dsigmoid(outputs[i]))
        db[i] = err
        dw[i] = err.dot(vals[i - 1].transpose())
    return dw, db


def train(t_data, h_data=None):
    # global epochs, w, b  # lol
    epoch = 0
    w = [np.array([0])] + [np.random.rand(x, y) for x, y in zip(neurons[1:], neurons[:-1])]
    b = np.array([np.random.randn(x, 1) for x in neurons])

    while epoch <= epochs and error_reasonable():
        np.random.shuffle(t_data)
        batches = [t_data[i:i + batch_size] for i in
                        range(0, len(t_data), batch_size)]

        for batch in batches:
            db = [np.zeros(s.shape) for s in b]
            dw = [np.zeros(s.shape) for s in w]
            for x, y in batch:
                outputs, vals = forward_propagate(x, w, b)
                ddw, ddb = back_propagate(outputs, vals, x, y, w, b)
                db = [nb + dnb for nb, dnb in zip(db, ddb)]
                dw = [nw + dnw for nw, dnw in zip(dw, ddw)]

            w = [s - (eta / batch_size) * dw for s, dw in
                            zip(w, dw)]
            b = [s - (eta / batch_size) * db for s, db in
                           zip(b, db)]

        if h_data:
            accuracy = holdout_predict(h_data, w, b) / len(h_data)
            print("Accuracy of {1} during epoch {0}.".format(epoch + 1, accuracy))
        else:
            print("On epoch {0}.".format(epoch))

        epoch += 1
    return w, b


def predict(d, w, b):
    outputs, vals = forward_propagate(d, w, b)
    # print(phi)
    # print(d)
    print(vals)
    print(np.argmax(vals[-1]))
    return np.argmax(vals[-1])


def holdout_predict(h_data, w, b):
    validation_results = [(predict(x, w, b) == y) for x, y in h_data]
    # for x, y in h_data:
        # print(predict(x))
    return sum(result for result in validation_results)

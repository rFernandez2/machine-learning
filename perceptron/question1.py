import numpy as np
import matplotlib.pyplot as plt

file_name = "./data/3Ddata.txt"

def read_data(file):
    with open(file, "r") as data_file:
        data = [[float(n) for n in line.split()[0:3]] for line in data_file]
    return data


def read_labels(file):
    with open(file, "r") as data_file:
        data = [int(line.split()[3]) for line in data_file]
    return data


def pca():
    data = read_data(file_name)
    print(data)
    labels = read_labels(file_name)
    center = np.mean(data, axis = 0)
    data = np.subtract(data, center)
    data = np.transpose(data)
    assert data.shape == (3, 500)
    covariance = np.cov(data)
    w, v = np.linalg.eigh(covariance)
    idx = w.argsort()[::-1]
    w =w[idx]
    v = v[:, idx]
    print(v)
    transform = np.hstack((v[0].reshape(3, 1), v[1].reshape(3, 1)))
    new_data = transform.T.dot(data)
    print(new_data.shape)
    print(labels)

    plt.scatter(new_data[0], new_data[1], c=labels, s=40)
    plt.show()


def k_nearest(data, k):
    neighbors = np.min()


def isomap():
    data = read_data(file_name)


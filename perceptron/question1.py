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


def center_data(data):
    center = np.mean(data, axis=0)
    data = np.subtract(data, center)
    return data


def pca():
    data = read_data(file_name)
    labels = read_labels(file_name)
    data = center_data(data)
    data = np.transpose(data)
    assert data.shape == (3, 500)
    covariance = np.cov(data)
    w, v = np.linalg.eigh(covariance)
    idx = w.argsort()[::-1]
    v = v[:, idx]
    v = v[:, :2]
    new_data = v.T.dot(data)
    plt.title("PCA 3Ddata")
    plt.scatter(new_data[0], new_data[1], c=labels, s=40)
    plt.show()


def construct_kNN(data, k):
    from scipy.spatial import distance as dst
    kNN_graph = []
    for c, pt in enumerate(data):
        distances = [dst.euclidean(point, pt) for point in data]
        idx = np.argsort(distances)
        idx = [int(idx[i + 1]) for i in range(k)]
        distances = [distances[i] for i in idx]
        kNN_graph.append((idx, distances))

    for c, node in enumerate(kNN_graph):
        for r, neigh in enumerate(node[0]):
            if c not in kNN_graph[neigh][0]:
                kNN_graph[neigh][0].append(c)
                kNN_graph[neigh][1].append(node[0][r])

    return kNN_graph


def floyd_warshall(delta_mat):
    sz = len(delta_mat)
    for k in range(sz):
        for i in range(sz):
            for j in range(sz):
                temp_dist = delta_mat[i][k] + delta_mat[k][j]
                if temp_dist < delta_mat[i][j]:
                    delta_mat[i][j] = temp_dist
    return delta_mat


def isomap(k):
    data = read_data(file_name)
    sz = len(data)
    labels = read_labels(file_name)
    kNN_graph = construct_kNN(data, k)
    delta_mat = [[float("inf") for x in range(sz)] for x in range(sz)]
    for i, node in enumerate(kNN_graph):
        delta_mat[i][i] = 0
        for j in range(len(node[0])):
            delta_mat[i][node[0][j]] = node[1][j]

    # make graph distances symmetric
    for i in range(sz):
        for j in range(sz):
            min_val = min(delta_mat[i][j], delta_mat[j][i])
            delta_mat[i][j] = min_val
            delta_mat[j][i] = min_val

    delta_mat = np.array(delta_mat)
    shortest_paths = np.array(floyd_warshall(delta_mat))
    assert (shortest_paths.transpose() == shortest_paths).all()
    shortest_paths = np.square(shortest_paths)
    n_inv = 1 / sz
    n_inv_mat = [[n_inv for x in range(sz)] for x in range(sz)]
    dim_center_mat = np.subtract(np.identity(sz), n_inv_mat)
    gram_mat = np.matmul(np.matmul(dim_center_mat, shortest_paths), dim_center_mat)
    gram_mat /= -2

    w, v = np.linalg.eig(gram_mat)
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]

    w = [abs(x) for x in w]
    q = np.sqrt(np.diag(w))
    new_data = v.dot(q)
    plt.scatter(new_data[:, 0], new_data[:, 1], c=labels, s=40)
    plt.title("Isomap 3Ddata")
    plt.show()
    return new_data

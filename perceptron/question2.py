from utility import *

M = 5

file_name = "./data/train35.digits"
label_file = "./data/train35.labels"


def train_perceptron():
    data = read_data(file_name)
    data = [(normalize_vector(v)) for v in data]
    labels = read_labels(label_file)
    d = len(data[0])
    w = [0 for x in range(d)]
    avg_errors = [0 for n in range(M)]

    for m in range(M):
        error_count = 0
        for i, x in enumerate(data):
            prediction = -1
            if np.dot(w, x) >= 0:
                prediction = 1
            if prediction == -1 and labels[i] == 1:
                error_count += 1
                w = np.add(w, x)
            elif prediction == 1 and labels[i] == -1:
                error_count += 1
                w = np.subtract(w, x)
        avg_errors[m] = error_count

    plot_errors(avg_errors)
    return w


def test_perceptron(w):
    f = open("test35.predictions", "w+")
    test_cases = read_data("./data/test35.digits")
    test_cases = [(normalize_vector(v)) for v in test_cases]
    for x in test_cases:
        if np.dot(w, x) >= 0:
            f.write("1\n")
        else:
            f.write("-1\n")
    f.close()


def run():
    w = train_perceptron()
    test_perceptron(w)
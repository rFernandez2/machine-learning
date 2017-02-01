from utility import *

M = 5
file_name = "./data/train01234.digits"
label_file = "./data/train01234.labels"


# Shuffle code written with help of StackOverflow
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def train_multi_perceptron(classes):
    from sklearn.utils import shuffle
    data = read_data(file_name)
    data = np.array([(normalize_vector(v)) for v in data])
    labels = np.array(read_labels(label_file))
    data, labels = shuffle(data, labels, random_state=0)

    d = len(data[0])
    w = [[0 for x in range(d)] for x in range(classes)]

    avg_errors = [0 for n in range(M)]

    for m in range(M):
        error_count = 0
        for i, x in enumerate(data):
            prediction = np.argmax([np.dot(weight, x) for weight in w])
            if prediction != labels[i]:
                error_count += 1
                w[prediction] = np.subtract(w[prediction], x / 2)
                w[labels[i]] = np.add(w[labels[i]], x / 2)

        avg_errors[m] = error_count

    plot_errors(avg_errors)
    return w


def test_multi_perceptron(w):
    f = open("test01234.predictions", "w+")
    test_cases = read_data("./data/test01234.digits")
    test_cases = [(normalize_vector(v)) for v in test_cases]
    for x in test_cases:
        prediction = np.argmax([np.dot(weight, x) for weight in w])
        f.write(str(prediction))
        f.write("\n")
    f.close()


def run():
    w = train_multi_perceptron(5)
    test_multi_perceptron(w)

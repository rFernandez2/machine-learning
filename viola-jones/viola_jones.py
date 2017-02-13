import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

num_imgs = 50
labels = []


def init_labels():
    print("Initializing Labels.")
    for x in range(0, num_imgs, 2):
        labels.append(1)
        labels.append(-1)

init_labels()


def integral_image(image):
    from skimage import color
    from skimage import io
    i = color.rgb2grey(io.imread(image))
    length = len(i)
    width = len(i[0])
    ii = np.zeros((width, length))
    s = np.zeros((width, length))
    for x, row in enumerate(i):
        for y, pt in enumerate(row):
            s[x, y] = s[x, y - 1] + i[x, y]
            ii[x, y] = ii[x - 1, y] + s[x, y]
    return ii


def integral_all_images():
    iimages = np.zeros((num_imgs, 64, 64))
    for x in range(0, num_imgs, 2):
        iimages[x] = integral_image("./data/faces/face" + str(x) + ".jpg")
        iimages[x + 1] = integral_image("./data/background/" + str(x) + ".jpg")
    return iimages


def two_rects(x, y, dx, dy):
    return [[x, y, int(x + dx / 2), y + dy, int(x + dx / 2), y, x + dx, y + dy],
            [x, y, x + dx, int(y + dy / 2), x, int(y + dy / 2), x + dx, y + dy]]


def all_features(size, stride, size_step):
    features = []
    for x in range(0, size, stride):
        for y in range(0, size, stride):
            for dx in range(2, size - x, size_step):
                for dy in range(2, size - y, size_step):
                    features.extend(two_rects(x, y, dx, dy))
    return features


iimages = integral_all_images()
feature_tbl = all_features(64, 4, 4)


def compute_feature(i, f):
    img = iimages[i]
    ft = feature_tbl[f]
    return img[ft[6], ft[7]] + 2 * img[ft[4], ft[5]] - img[ft[6], ft[1]] - 2 * img[ft[2], ft[3]] - img[ft[0], ft[1]] + img[ft[0], ft[3]]


def find_all_sigma():
    print("Initializing Sigma (might take a while)")
    length = len(feature_tbl)
    sigma = []
    for f in range(length):
        sigma.append(np.argsort([compute_feature(i, f) for i in range(num_imgs)]))
    return sigma


sigmas = find_all_sigma()


def weak_classifier(w, f, imgs):
    sigma = sigmas[f]
    num_img = len(imgs)
    s_plus = np.zeros(num_img)
    s_minus = np.zeros(num_img)
    # print(sigma)

    if labels[sigma[0]] == 1:
        s_plus[0] = w[sigma[0]]
        s_minus[0] = 0
    else:
        s_plus[0] = 0
        s_minus[0] = w[sigma[0]]

    for i in range(1, num_img):
        if labels[sigma[i]] == 1:
            s_plus[i] = s_plus[i - 1] + w[sigma[i]]
            s_minus[i] = s_minus[i - 1]
        else:
            s_plus[i] = s_plus[i - 1]
            s_minus[i] = s_minus[i - 1] + w[sigma[i]]
    t_plus = s_plus[num_img - 1]
    t_minus = s_minus[num_img - 1]
    errors = np.minimum(np.add(s_plus, np.subtract(t_minus, s_minus)), np.add(s_minus, np.subtract(t_plus, s_plus)))
    min_idx = np.argmin(errors)
    theta = 0
    if min_idx == num_img - 1:
        theta = compute_feature(sigma[min_idx], f)
    else:
        theta = (compute_feature(sigma[min_idx + 1], f) - compute_feature(sigma[min_idx], f)) / 2
    if s_plus[min_idx] + (t_minus - s_minus[min_idx]) > s_minus[min_idx] + (t_plus - s_plus[min_idx]):
        polarity = -1
    else:
        polarity = 1
    return [f, polarity, theta]


def best_learner(w, classifiers, imgs):
    errors = []
    for f, p, t in classifiers:
        vals = [1 if p * compute_feature(i, f) < p * t else -1 for i in imgs]
        errors.append(np.sum(np.multiply(w, np.absolute(np.subtract(vals, labels)))))

    idx = np.argmin(errors)

    print(classifiers[idx])
    return classifiers[idx], errors[idx]


def compute_strong(h, i):
    sum1 = 0
    sum2 = 0
    for classifier in h[0]:
        if classifier[2] * compute_feature(i, classifier[1]) < classifier[2] * classifier[3]:
            sum1 += classifier[0]
        else:
            sum1 -= classifier[0]
        sum2 += classifier[0] / 2
    return [1 if sum1 > sum2 else -1, sum1]


def train_single(w, imgs):
    print("Starting round of boosting")
    classifiers = [weak_classifier(w, f, imgs) for f in range(len(feature_tbl))]
    best_classifier, error = best_learner(w, classifiers, imgs)
    beta = error / (1 - error)
    alpha = np.log(1 / beta)
    return [alpha, best_classifier[0], best_classifier[1], best_classifier[2]], beta


def cascade_training():
    num_features = len(feature_tbl)
    num_imgs = len(iimages)
    imgs = list(range(num_imgs))
    final_classifier = []
    w = np.full(num_imgs, 1 / num_imgs)
    non_faces = 1
    t = 0
    false_neg = 1
    false_pos = 1
    while non_faces > 0.01:
        strong_hypothesis = [[], 0]
        hypotheses = []
        while false_neg != 0 or false_pos > 0.3 * len(imgs):
            false_pos = 0
            false_neg = 0
            sum_w = np.sum(w)
            w /= sum_w
            print(w)
            c, beta = train_single(w, imgs)
            strong_hypothesis[0].append(c)
            print("Calculating hypotheses")
            hypotheses = [compute_strong(strong_hypothesis, i) for i in imgs]
            non_faces = 0
            theta = 99999999
            for idx, i in enumerate(imgs):
                if labels[i] == 1:
                    if hypotheses[idx][0] == 1:
                        w[i] *= beta
                    else:
                        false_neg += 1
                        theta = min(theta, hypotheses[idx][1])
                else:
                    if hypotheses[idx][0] == -1:
                        w[i] *= beta
                    else:
                        false_pos += 1
            if false_neg == 0:
                theta = 0
            print("False negatives: " + str(false_neg) + "\t False Positives: " + str(false_pos))
            print(strong_hypothesis)
            strong_hypothesis[1] = theta

        # RECALCULATE HYPOTHESES HERE
        final_classifier.append(strong_hypothesis)
        for idx, i in enumerate(imgs):
            if hypotheses[idx][0] == -1 and labels[i] == -1:
                imgs.remove(i)

    return final_classifier

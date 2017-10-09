import numpy as np
import matplotlib.pyplot as plt

num_i = 800
num_imgs = 1600


def integral_image(i):
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
    from skimage import io
    images = []
    label = []
    print("Calculating integral images.")
    for x in range(num_i):
        i = io.imread("./data/faces/face" + str(x) + ".jpg", as_grey=True)
        images.append(integral_image(i))
        label.append(1)
        i = io.imread("./data/background/" + str(x) + ".jpg", as_grey=True)
        images.append(integral_image(i))
        label.append(-1)
    return images, label


def find_all_sigma(img):
    print("Initializing Sigma (might take a while)")
    length = len(feature_tbl)
    sigma = []
    for f in range(length):
        sigma.append(np.argsort([compute_feature(i, f) for i in img]))
    return sigma


def two_rects(x, y, dx, dy):
    return [([x, y, int(x + dx / 2), y + dy, int(x + dx / 2), y, x + dx, y + dy], 0),
            ([x, y, x + dx, int(y + dy / 2), x, int(y + dy / 2), x + dx, y + dy], 1)]


def three_rects(x, y, dx, dy):
    return [([x, y, int(x + dx / 3), y + dy, int(x + 2 * dx / 3), y, x + dx, y + dy], 2)]


def four_rects(x, y, dx, dy):
    return [([x, y, int(x + dx / 2), int(y + dy / 2), x + dx, y + dy, x, y + dy], 3)]


def all_features(size, stride, size_step, triple_step):
    features = []
    print("Populating feature table.")
    for x in range(0, size, stride):
        for y in range(0, size, stride):
            for dx in range(2, size - x, size_step):
                for dy in range(2, size - y, size_step):
                    features.extend(two_rects(x, y, dx, dy))
                    features.extend(four_rects(x, y, dx, dy))

    for x in range(0, size, stride):
        for y in range(0, size, stride):
            for dx in range(3, size - x, triple_step):
                for dy in range(2, size - y, size_step):
                    features.extend(three_rects(x, y, dx, dy))
    return features


iimages, labels = integral_all_images()
feature_tbl = all_features(64, 4, 4, 3)


def compute_feature(i, f):
    img = iimages[i]
    ft = feature_tbl[f][0]
    flag = feature_tbl[f][1]
    if flag == 0:
        return img[ft[6], ft[7]] + 2 * img[ft[4], ft[5]] - img[ft[6], ft[1]] - 2 * img[ft[2], ft[3]] - img[ft[0], ft[1]] + img[ft[0], ft[3]]
    elif flag == 1:
        return img[ft[6], ft[7]] + 2 * img[ft[4], ft[5]] - img[ft[4], ft[7]] - 2 * img[ft[2], ft[3]] - img[ft[0], ft[1]] + img[ft[0], ft[3]]
    elif flag == 2:
        return img[ft[6], ft[7]] + 2 * img[ft[4], ft[5]] - img[ft[6], ft[5]] - 2 * img[ft[4], ft[3]] + 2 * img[ft[2], ft[3]] + img[ft[0], ft[1]] - 2 * img[ft[2], ft[1]] - img[ft[0], ft[3]]
    else:
        return img[ft[6], ft[7]] + img[ft[4], ft[5]] + 4 * img[ft[2], ft[3]] - 2 * img[ft[4], ft[3]] - 2 * img[ft[2], ft[5]] + img[ft[0], ft[1]] - 2 * img[ft[2], ft[1]] - 2 * img[ft[0], ft[3]] + img[ft[4], ft[1]]


def best_learner(w, imgs, sigmas):
    best_f = [99999, 0, 0, 0]
    num_img = len(imgs)
    for f in range(len(feature_tbl)):
        sigma = sigmas[f]
        s_plus = np.zeros(num_img)
        s_minus = np.zeros(num_img)

        if labels[imgs[sigma[0]]] == 1:
            s_plus[0] = w[sigma[0]]
            s_minus[0] = 0
        else:
            s_plus[0] = 0
            s_minus[0] = w[sigma[0]]

        for i in range(1, num_img):
            if labels[imgs[sigma[i]]] == 1:
                s_plus[i] = s_plus[i - 1] + w[sigma[i]]
                s_minus[i] = s_minus[i - 1]
            else:
                s_plus[i] = s_plus[i - 1]
                s_minus[i] = s_minus[i - 1] + w[sigma[i]]
        t_plus = s_plus[num_img - 1]
        t_minus = s_minus[num_img - 1]
        left_min = np.add(s_plus, np.subtract(t_minus, s_minus))
        right_min = np.add(s_minus, np.subtract(t_plus, s_plus))
        errors = np.minimum(left_min, right_min)

        min_idx = np.argmin(errors)

        if errors[min_idx] < best_f[0]:
            theta = 0
            if min_idx == num_img - 1:
                theta = compute_feature(imgs[sigma[min_idx]], f)
            else:
                theta = (compute_feature(imgs[sigma[min_idx + 1]], f) + compute_feature(imgs[sigma[min_idx]], f)) / 2
            if s_plus[min_idx] + (t_minus - s_minus[min_idx]) > s_minus[min_idx] + (t_plus - s_plus[min_idx]):
                polarity = -1
            else:
                polarity = 1
            best_f = [errors[min_idx], f, polarity, theta]
    print("BEST LEARNER: " + str(best_f[1]) + "\tWITH ERROR: " + str(best_f[0]))
    return best_f[1:], best_f[0]


def compute_strong(h, i):
    sum1 = 0
    for classifier in h[0]:
        if np.sign(classifier[2] * (compute_feature(i, classifier[1]) - classifier[3])) >= 0:
            sum1 += classifier[0]
        else:
            sum1 -= classifier[0]
    return [1 if sum1 - h[1] >= 0 else -1, sum1]


def compute_cascade(cascade, i):
    for classifier in cascade:
        if compute_strong(classifier, i)[0] == -1:
            return -1
    return 1


def compute_weak(h, i):
    if np.sign(h[1] * (compute_feature(i, h[0]) - h[2])) >= 0:
        return 1
    else:
        return -1


def cascade_training():
    imgs = list(range(num_imgs))
    final_classifier = []
    non_faces = num_i
    faces = num_i
    while non_faces > 0.01 * num_i:
        print("NEW CASCADE LEARNER")
        num_img = len(imgs)
        strong_hypothesis = [[], 0]
        false_pos =  num_i
        w = [1. / (2 * faces) if labels[i] == 1 else 1. / (2 * non_faces) for i in imgs]
        sigmas = find_all_sigma(imgs)

        while false_pos > 0.4 * non_faces:
            false_pos = 0
            false_neg = 0
            sum_w = np.sum(w)
            w /= sum_w

            print("Starting round of boosting")
            best_classifier, error = best_learner(w, imgs, sigmas)
            beta = error / (1 - error)
            alpha = np.log(1 / beta)
            strong_hypothesis[0].append([alpha, best_classifier[0], best_classifier[1], best_classifier[2]])

            hypotheses = [compute_weak([best_classifier[0], best_classifier[1], best_classifier[2]], i) for i in imgs]

            for idx, i in enumerate(imgs):
                if labels[i] == 1 and hypotheses[idx] >= 0:
                    w[idx] *= beta
                elif labels[i] == -1 and hypotheses[idx] == -1:
                    w[idx] *= beta

            theta = min([compute_strong(strong_hypothesis, i)[1] if labels[i] == 1 else 0 for i in imgs])
            if theta > 0:
                theta = 0

            strong_hypothesis[1] = theta
            hypotheses = [compute_strong(strong_hypothesis, i) for i in imgs]
            for idx, i in enumerate(imgs):
                if labels[i] == -1 and hypotheses[idx][0] >= 0:
                    false_pos += 1
                elif labels[i] == 1 and hypotheses[idx][0] == -1:
                    false_neg += 1
            assert false_neg == 0

            print("False Positives: " + str(false_pos) + " out of " + str(len(imgs)))

        final_classifier.append(strong_hypothesis)
        hypotheses = [compute_strong(strong_hypothesis, i) for i in imgs]

        imgs[:] = [i for idx, i in enumerate(imgs) if not (hypotheses[idx][0] == -1 and labels[i] == -1)]
        non_faces -= (num_img - len(imgs))
        print("NON_FACES= " + str(non_faces))

    return final_classifier


def detect_feature(img, f):
    ft = feature_tbl[f][0]
    flag = feature_tbl[f][1]
    if flag == 0:
        return img[ft[6], ft[7]] + 2 * img[ft[4], ft[5]] - img[ft[6], ft[1]] - 2 * img[ft[2], ft[3]] - img[ft[0], ft[1]] + img[ft[0], ft[3]]
    else:
        return img[ft[6], ft[7]] + 2 * img[ft[4], ft[5]] - img[ft[4], ft[7]] - 2 * img[ft[2], ft[3]] - img[ft[0], ft[1]] + img[ft[0], ft[3]]


def detect_strong(h, ii, gamma):
    sum1 = 0
    sum2 = 0
    for classifier in h:
        sum2 += classifier[0]
        if np.sign(classifier[2] * (detect_feature(ii, classifier[1]) - classifier[3])) >= 0:
            sum1 += classifier[0]
        else:
            sum1 -= classifier[0]
    if sum1 >= gamma * sum2:
        return [1, sum1]
    else:
        return [-1, 0]


def test_cascade(cascade, img, gamma):
    hyp = []
    for classifier in cascade:
        hyp = detect_strong(classifier, img, gamma)
        if hyp[0] == -1:
            return [-1, 0]
    return [1, hyp[1]]


def rect_overlaps(rects, x1, y1):
    x2 = x1 + 64
    y2 = y1 + 64
    overlaps = 0
    for rect in rects:
        rx1 = rect[0]
        ry1 = rect[1]
        rx2 = rx1 + 64
        ry2 = ry1 + 64
        if rx1 <= x1 <= rx2 and (ry1 <= y1 <= ry2 or ry1 <= y2 <= ry2):
            overlaps += 1
        elif rx1 <= x2 <= rx2 and (ry1 <= y1 <= ry2 or ry1 <= y2 <= ry2):
            overlaps += 1
    return overlaps


def detect_image(cascade, gamma, overlaps):
    from matplotlib import patches
    from skimage import io
    i = io.imread("./data/class.jpg", as_grey=True)
    length = len(i)
    width = len(i[0])
    faces = []
    vals = []

    y = 0
    x = 0
    while y + 64 < width:
        x = 0
        while x + 64 < length:
            img = integral_image(i[x:(x + 64), y:(y + 64)])
            c = test_cascade(cascade, img, gamma)
            if c[0] == 1:
                vals.append([x, y, c[0], c[1]])
            x += 2
        y += 2
    fig, ax = plt.subplots(1)
    io.imshow(i)
    vals = sorted(vals, key=lambda val_entry: val_entry[3])
    rects = []
    if overlaps == 1:
        for val in vals:
            if rect_overlaps(rects, val[0], val[1]) == 0:
                faces.append(patches.Rectangle((val[1], val[0]), 64, 64, linewidth=2, edgecolor='r', facecolor='none'))
                rects.append([val[0], val[1]])

    else:
        for val in vals:
            faces.append(patches.Rectangle((val[1], val[0]), 64, 64, linewidth=2, edgecolor='r', facecolor='none'))

    for face in faces:
        ax.add_patch(face)

    plt.show()
    return vals


# gamma is desired gamma value
# overlaps == 1 means no overlap overlaps == 0 means show all
def train_and_test(gamma, overlaps):
    h = cascade_training()
    h = [h[i][0] for i in range(len(h))]
    detect_strong(h, gamma, overlaps)
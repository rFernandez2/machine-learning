import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

num_i = 500
num_imgs = 1000
very_big_number = 9999999


def integral_image(image):
    from skimage import color
    from skimage import io
    import skimage.transform as trans
    i = color.rgb2grey(io.imread(image))
    length = len(i)
    width = len(i[0])
    iii = trans.integral_image(i)

    ii = np.zeros((width, length))
    s = np.zeros((width, length))
    for x, row in enumerate(i):
        for y, pt in enumerate(row):
            s[x, y] = s[x, y - 1] + i[x, y]
            ii[x, y] = ii[x - 1, y] + s[x, y]
    # assert (ii == iii).all()
    return iii


def integral_all_images():
    iimages = []
    labels = []
    print("Calculating integral images.")
    for x in range(num_i):
        iimages.append(integral_image("./data/faces/face" + str(x) + ".jpg"))
        labels.append(1)
        iimages.append(integral_image("./data/background/" + str(x) + ".jpg"))
        labels.append(-1)
    return iimages, labels


def two_rects(x, y, dx, dy):
    return [([x, y, int(x + dx / 2), y + dy, int(x + dx / 2), y, x + dx, y + dy], 0),
            ([x, y, x + dx, int(y + dy / 2), x, int(y + dy / 2), x + dx, y + dy], 1)]


def all_features(size, stride, size_step):
    features = []
    print("Populating feature table.")
    for x in range(0, size, stride):
        for y in range(0, size, stride):
            for dx in range(2, size - x, size_step):
                for dy in range(2, size - y, size_step):
                    features.extend(two_rects(x, y, dx, dy))
    return features


iimages, labels = integral_all_images()
feature_tbl = all_features(64, 4, 4)


def compute_feature(i, f):
    img = iimages[i]
    ft = feature_tbl[f][0]
    flag = feature_tbl[f][1]
    if flag == 0:
        return img[ft[6], ft[7]] + 2 * img[ft[4], ft[5]] - img[ft[6], ft[1]] - 2 * img[ft[2], ft[3]] - img[ft[0], ft[1]] + img[ft[0], ft[3]]
    else:
        return img[ft[6], ft[7]] + 2 * img[ft[4], ft[5]] - img[ft[4], ft[7]] - 2 * img[ft[2], ft[3]] - img[ft[0], ft[1]] + img[ft[0], ft[3]]


def find_all_sigma(img):
    print("Initializing Sigma (might take a while)")
    length = len(feature_tbl)
    sigma = []
    for f in range(length):
        sigma.append(np.argsort([compute_feature(i, f) for i in img]))
    return sigma


'''
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
        polarity = 1
    else:
        polarity = -1
    return [f, polarity, theta]
'''


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
                theta = compute_feature(sigma[min_idx], f)
            else:
                theta = (compute_feature(sigma[min_idx + 1], f) + compute_feature(sigma[min_idx], f)) / 2
            if s_plus[min_idx] + (t_minus - s_minus[min_idx]) > s_minus[min_idx] + (t_plus - s_plus[min_idx]):
                polarity = -1
            else:
                polarity = 1
            best_f = [errors[min_idx], f, polarity, theta]

    hypotheses = [compute_feature(i, best_f[1]) for i in range(num_img)]
    #print(hypotheses)
    return best_f[1:], best_f[0]


def compute_strong(h, i):
    sum1 = 0
    for classifier in h[0]:
        if np.sign(classifier[2] * (compute_feature(i, classifier[1]) - classifier[3])) >= 0:
            sum1 += classifier[0]
        else:
            sum1 -= classifier[0]
    return [1 if sum1 - h[1] >= 0 else -1, sum1]


def cascade_training():
    imgs = list(range(num_imgs))
    final_classifier = []
    non_faces = num_i
    while non_faces > 0.03 * num_imgs:
        print("NEW CASCADE LEARNER")
        num_img = len(imgs)
        sigmas = find_all_sigma(imgs)
        strong_hypothesis = [[], 0]
        hypotheses = []
        false_neg = 1
        false_pos = 1
        w = np.full(num_img, 1 / num_img)
        while false_neg != 0 or false_pos > 0.3 * non_faces:
            false_pos = 0
            false_neg = 0
            sum_w = np.sum(w)
            w /= sum_w
            #print(w)
            print("Starting round of boosting")
            best_classifier, error = best_learner(w, imgs, sigmas)
            beta = error / (1 - error)
            alpha = np.log(1 / beta)
            strong_hypothesis[0].append([alpha, best_classifier[0], best_classifier[1], best_classifier[2]])

            # print([compute_feature(i, 15000) for i in imgs])
            # print(hypotheses)
            theta = very_big_number
            hypotheses = [compute_strong(strong_hypothesis, i) for i in imgs]

            for idx, i in enumerate(imgs):
                if labels[i] == 1:
                    theta = min(theta, hypotheses[idx][1])
                    if hypotheses[idx][0] >= 0:
                        w[idx] *= beta
                else:
                    # print("NEGATIVE: " + str(hypotheses[idx][1]))
                    if hypotheses[idx][0] == -1:
                        w[idx] *= beta

            if theta == very_big_number:
                theta = 0
            strong_hypothesis[1] = theta
            hypotheses = [compute_strong(strong_hypothesis, i) for i in imgs]
            for idx, i in enumerate(imgs):
                if labels[i] == -1 and hypotheses[idx][0] >= 0:
                    false_pos += 1
                elif labels[i] == 1 and hypotheses[idx][0] == -1:
                    false_neg += 1
            assert false_neg == 0

            # print(hypotheses)

            print("AFTER THETA CHANGE: False negatives: " + str(false_neg) + "\t False Positives: " + str(false_pos) + " out of " + str(len(imgs)))
            #print(imgs)
            #END DEBUGGING
            print(strong_hypothesis)

            '''
            #BOTTOM IS FOR TESTING
            false_pos2 = 0
            false_neg2 = 0
            # print(strong_hypothesis)
            hypotheses2 = [compute_strong(strong_hypothesis, i) for i in imgs]
            for idx, i in enumerate(imgs):
                if labels[i] == 1:
                    if hypotheses2[idx][0] == 1:
                        w[i] *= beta
                    else:
                        false_neg2 += 1
                    theta = min(theta, hypotheses[idx][1])
                    # print(theta)
                else:
                    if hypotheses2[idx][0] == -1:
                        w[i] *= beta
                    else:
                        false_pos2 += 1
            print("2: False negatives: " + str(false_neg2) + "\t False Positives: " + str(false_pos2))
            #TESTING CODE ENDS HERE
            '''

        # RECALCULATE HYPOTHESES HERE
        final_classifier.append(strong_hypothesis)
        hypotheses = [compute_strong(strong_hypothesis, i) for i in imgs]

        imgs[:] = [i for idx, i in enumerate(imgs) if not (hypotheses[idx][0] == -1 and labels[i] == -1)]
        non_faces -= (num_img - len(imgs))
        print("NON_FACES= " + str(non_faces))

    return final_classifier


def calc_integral(i):
    length = len(i)
    width = len(i[0])
    ii = np.zeros((width, length))
    s = np.zeros((width, length))
    for x, row in enumerate(i):
        for y, pt in enumerate(row):
            s[x, y] = s[x, y - 1] + i[x, y]
            ii[x, y] = ii[x - 1, y] + s[x, y]
    return ii


'''
def detect_image(cascade):
    from skimage import color
    from skimage import io
    i = color.rgb2grey(io.imread("./data/class.jpg"))
    length = len(i)
    width = len(i[0])
    for x in range(0, length, 64):
        for y in range(0, width, 64):
            img = calc_integral(i[x:x+64][y:y+64])
            sum1 = 0
            for classifier in cascade:
                if np.sign(classifier[2] * (compute_feature(i, classifier[1]) - classifier[3])) >= 0:
                    sum1 += classifier[0]
                else:
                    sum1 -= classifier[0]
            np.sign(sum1 - h[1])
'''
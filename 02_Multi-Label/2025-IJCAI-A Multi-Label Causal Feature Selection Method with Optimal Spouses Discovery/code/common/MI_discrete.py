import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

def cmi(train_label, feature1, feature2):

    train_label = train_label.astype(int)
    feature1 = feature1.astype(int)
    feature2 = feature2.astype(int)

    joint_entropy = stats.entropy(np.ravel_multi_index((train_label, feature1, feature2), (train_label.max()+1, feature1.max()+1, feature2.max()+1)))

    conditional_entropy = stats.entropy(np.ravel_multi_index((train_label, feature1), (train_label.max()+1, feature1.max()+1)))

    iXYZ = joint_entropy - conditional_entropy

    return iXYZ

def mi(firstVector, secondVector):
    iXY = mutual_info_score(firstVector, secondVector)
    return iXY


def MI_G2(Data, target, alpha, p):

    ntest = 0
    mb = []

    train_data = Data[:, np.setdiff1d(range(p), target)]
    featureMatrix = train_data

    train_label = Data[:, target]
    classColumn = train_label

    numFeatures = featureMatrix.shape[1]
    classScore = np.zeros(numFeatures)
    vis = np.zeros(p)

    for i in range(numFeatures):
        ntest += 1
        classScore[i] = SU(featureMatrix[:, i], classColumn)

    classScore, indexScore = zip(*sorted(zip(classScore, range(numFeatures)), reverse=True))
    threshold = 0.05
    t = [index for index, score in zip(indexScore, classScore) if score < threshold]
    u = [score for score in classScore if score < threshold]
    indexScore = [index for index, score in zip(indexScore, classScore) if score > threshold]
    classScore = [score for score in classScore if score > threshold]

    if len(indexScore) == 0:
        selectedFeatures = []
        return mb, ntest, 0

    curPosition = 0
    mii = -1

    while curPosition < len(indexScore):
        mb_tmp = []
        j = curPosition + 1
        curFeature = indexScore[curPosition]
        while j < len(indexScore):
            scoreij = SU(featureMatrix[:, curFeature], featureMatrix[:, indexScore[j]])
            ntest += 1
            if scoreij > classScore[j]:
                indexScore.pop(j)
                classScore.pop(j)
            else:
                j += 1
        curPosition += 1

    selectedFeatures = indexScore
    pc = selectedFeatures
    last = selectedFeatures[-1]
    mb = yingshe2(pc, target)
    for feature in selectedFeatures:
        vis[feature] = 1

    len1 = len(selectedFeatures)
    for i in range(len1):
        mb_tmp = []
        a = selectedFeatures.index(last) + 1
        len2 = len(t)
        while a < len2:
            if vis[t[a]] == 1:
                a += 1
                continue
            scoreij = SU(featureMatrix[:, selectedFeatures[i]], featureMatrix[:, t[a]])
            ntest += 1
            if scoreij > u[a] + 0.13:
                iXYZ = cmi(train_label, featureMatrix[:, t[a]], featureMatrix[:, selectedFeatures[i]])
                iXY0 = mi(train_label, featureMatrix[:, t[a]])
                ntest += 2
                if t[a] >= target:
                    ttt = t[a] + 1
                else:
                    ttt = t[a]
                if iXYZ > iXY0:
                    mb_tmp.append(t[a])
                    mb.append(ttt)
                    vis[t[a]] = 1
                if t[a] in indexScore:
                    indexScore.remove(t[a])
                    classScore.remove(u[a])
            a += 1

    time = 0
    return mb, ntest, time

def SU(firstVector, secondVector):
    hX = h(firstVector)
    hY = h(secondVector)
    iXY = mi(firstVector, secondVector)
    score = (2 * iXY) / (hX + hY)
    return score

def h(vector):
    _, counts = np.unique(vector, return_counts=True)
    probabilities = counts / len(vector)
    return entropy(probabilities, base=2)

def yingshe2(pc, target):
    pc = [p if p < target else p + 1 for p in pc]
    return pc

def mutual_information(firstVector, secondVector):

    joint_entropy = entropy(np.vstack((firstVector, secondVector)))
    marginal_entropy_x = entropy(firstVector)
    marginal_entropy_y = entropy(secondVector)
    mi1 = marginal_entropy_x + marginal_entropy_y - joint_entropy
    return mi1

def conditional_mutual_information(data_matrix, x, y, s):

    first_variable = np.concatenate((data_matrix[:, x].reshape(-1, 1), data_matrix[:, s]), axis=1)
    second_variable = np.concatenate((data_matrix[:, y].reshape(-1, 1), data_matrix[:, s]), axis=1)
    cmi1 = mutual_information(first_variable, second_variable)
    return cmi1

def mutual_information_test(data, x, y, s):

    mi_xy = mi(data[:, x], data[:, y])
    if not s:
        return 1, mi_xy

    mi_xys = conditional_mutual_information(data, x, y, s)

    p_value = 1 if (mi_xy >= mi_xys).any() else 0

    return p_value, abs(mi_xy - mi_xys)
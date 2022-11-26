import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import tree
import matplotlib.pyplot as plt

data = np.array(pd.read_csv('./heart_disease_health_indicators_BRFSS2015.csv', header = 0, usecols = ['HeartDiseaseorAttack','HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','Diabetes','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost', 'GenHlth', 'MentHlth','PhysHlth','DiffWalk', 'Age', 'Income']).values)

high_income = data[np.where(data[:, -1] >= 8)]  # source domain
low_income = data[np.where(data[:, -1] <= 1)]  # target domain

def down_sampling(data):  # down sample to make source domain balance
    pos_dp = []
    neg_dp = []
    for dp in data:
        if dp[0] == 1:
            pos_dp.append(dp)
        else:
            neg_dp.append(dp)

    pos_dp = np.array(pos_dp)
    neg_dp = np.array(neg_dp)
    idx = np.random.randint(len(neg_dp), size=len(pos_dp))
    choice = data[idx,:]
    data = np.vstack((pos_dp, choice))
    np.random.shuffle(data)
    return data

high_income = down_sampling(high_income)

X_s = high_income[:, 1:(len(high_income) - 1)]  # drop Income column
y_s = high_income[:, 0]

X_t = low_income[:, 1:(len(high_income) - 1)]
y_t = low_income[:, 0]

print(len(X_s))
print(len(X_t))

def standardization_normalization(column_data, method="standardization"):
    if method == "standardization":
        mu = np.mean(column_data)
        std = np.std(column_data)
        column_data = (column_data - mu) / std
        return column_data

    elif method == "normalization":
        max_value = max(column_data)
        min_value = min(column_data)
        column_data = (column_data - min_value) / (max_value - min_value)
        return column_data
    else:
        return column_data

for i in [3, 13, 14, 15, 17]:  # BMI, GenHlth, MentHlth, PhysHlth, Age columns
    method="None"
    X_s[:, i] = standardization_normalization(X_s[:, i], method=method)
    X_t[:, i] = standardization_normalization(X_t[:, i], method=method)

X_s_train, X_s_val, y_s_train, y_s_val = train_test_split(X_s, y_s, test_size=0.3)

X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(X_t, y_t, test_size=0.8)

def mix_data(X_s_train, y_s_train, X_t_train, y_t_train, target_data_percentage):
    n = int(target_data_percentage / 100 * len(X_t_train))
    idx = np.random.randint(len(X_t_train), size=n)
    choice_X = X_t_train[idx,:]
    choice_y = y_t_train[idx]
    X_s_train = np.vstack((X_s_train, choice_X))
    y_s_train = np.hstack((y_s_train, choice_y))
    return X_s_train, y_s_train

X_s_train, y_s_train = mix_data(X_s_train, y_s_train, X_t_train, y_t_train, 0)

def get_accuracy(predicted_labels, true_labels):
    n = len(true_labels)
    count = 0
    for i in range(n):
        if predicted_labels[i] == true_labels[i]:
            count += 1
    return count / n

def trivial_system(y_t_test):
    n = len(y_t_test)
    predicted_labels = np.random.randint(2, size=n)
    print(get_accuracy(predicted_labels, y_t_test))

trivial_system(y_t_test)

def logistic_regression(X_s_train, y_s_train, X_s_val, y_s_val, X_t_test, y_t_test):
    max_iter = 100
    clf = LogisticRegression(solver='saga', max_iter=max_iter).fit(X_s_train, y_s_train)
    score = clf.score(X_t_test, y_t_test)

    clf = LogisticRegression(solver='saga', penalty='l1', max_iter=max_iter).fit(X_s_train, y_s_train)
    score_l1 = clf.score(X_t_test, y_t_test)

    clf = LogisticRegression(solver='saga', penalty='l2', max_iter=max_iter).fit(X_s_train, y_s_train)
    score_l2 = clf.score(X_t_test, y_t_test)
    return [score, score_l1, score_l2]

logistic_score = []
logistic_score_l1 = []
logistic_score_l2 = []

for mix_percent in range(0, 101, 5):
    print(mix_percent)
    X_s_train_mix, y_s_train_mix = mix_data(X_s_train, y_s_train, X_t_train, y_t_train, mix_percent)
    score_list = logistic_regression(X_s_train_mix, y_s_train_mix, X_s_val, y_s_val, X_t_test, y_t_test)
    print(score_list)

    logistic_score.append(score_list[0])
    logistic_score_l1.append(score_list[1])
    logistic_score_l2.append(score_list[2])

plt.figure()
num_data_mixed = [i/100*0.2*len(X_t) for i in range(0, 101, 5)]
plt.plot(num_data_mixed, logistic_score, label="Logistic regression")
plt.plot(num_data_mixed, logistic_score_l1, label="Logistic regression with L1 regularization")
plt.plot(num_data_mixed, logistic_score_l2, label="Logistic regression with L2 regularization")
plt.xlabel("num of data in target domain mixed")
plt.ylabel("test accuracy")
plt.legend()
plt.show()

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_s_train, y_s_train)
print(clf.score(X_t_test, y_t_test))

clf = tree.DecisionTreeClassifier()
clf.fit(X_s_train, y_s_train)
print(clf.score(X_t_test, y_t_test))

def get_sign_matrix(length):
    end = 2 ** length - 1
    start = 0
    res = []
    while start <= end:
        cur = start
        cur_bits = []
        while cur > 0:
            cur_bits.append(cur & 1)
            cur = cur >> 1
        while len(cur_bits) < length:
            cur_bits.append(0)
        for i in range(len(cur_bits)):
            if cur_bits[i] == 0:
                cur_bits[i] = -1
        res.append(cur_bits)
        start += 1
    return res

def subspace_alignment(Xs, Ys, Xt_test, yt_test, clf, X_t_train=None, y_t_train=None, num_features=19):
    source_pca = PCA(num_features)
    source_pca.fit(Xs)

    target_pca = PCA(num_features)
    target_pca.fit(Xt_test)

    M = source_pca.components_ @ target_pca.components_.T
    Xs_tf = source_pca.transform(Xs) @ M
    clf.fit(Xs_tf, Ys)
    
    if X_t_train is None or y_t_train is None:
        Xt_tf = target_pca.transform(Xt_test)
        score = clf.score(Xt_tf, yt_test)
    else:
        sign_martix = None
        bestacc = 0
        for diag in get_sign_matrix(num_features):
            cur_sign_martix = np.zeros((num_features, num_features), int)
            np.fill_diagonal(cur_sign_martix, diag)

            Xt_tf = target_pca.transform(X_t_train) @ cur_sign_martix
            acc = clf.score(Xt_tf, y_t_train)
            if acc > bestacc:
                bestacc = acc
                sign_martix = cur_sign_martix

        Xt_test_tf = target_pca.transform(Xt_test) @ sign_martix
        score = clf.score(Xt_test_tf, yt_test)
    print(score)
    return score

plt.figure()
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
acc_list = []

N = 11
# for i in range(1, len(X_s_train[0])):
for i in range(1, N):
    print(i)
    acc_list.append(subspace_alignment(X_s_train, y_s_train, X_t_test, y_t_test, clf, num_features=i))
plt.plot(range(1, len(acc_list) + 1, 1), acc_list, label="SVM without target domain data")

for N_tl in [10, 100, 1000, 2000, 3000, 4000]:
    acc_list_ntl = []
    for i in range(1, N):
        X_tl = np.concatenate((X_t_train[:N_tl//2],X_t_train[-N_tl//2:]), axis=0)
        y_tl = np.concatenate((y_t_train[:N_tl//2],y_t_train[-N_tl//2:]), axis=0)
        acc_list_ntl.append(subspace_alignment(X_s_train, y_s_train, X_t_test, y_t_test, clf, X_t_train=X_tl, y_t_train=y_tl, num_features=i))
    plt.plot(range(1, len(acc_list_ntl) + 1, 1), acc_list_ntl, label="SVM with " + str(N_tl) + " target domain data")

plt.xlabel("num of features kept")
plt.ylabel("accuracy")
plt.legend()
plt.show()

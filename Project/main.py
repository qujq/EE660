import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import tree

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
    method="standardization"
    X_s[:, i] = standardization_normalization(X_s[:, i], method=method)
    X_t[:, i] = standardization_normalization(X_t[:, i], method=method)

X_s_train, X_s_val, y_s_train, y_s_val = train_test_split(X_s, y_s, test_size=0.3)

X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(X_t, y_t, test_size=0.3)

def mix_data(X_s_train, y_s_train, X_t_train, y_t_train, target_data_percentage):
    n = int(target_data_percentage / 100 * len(X_t_train))
    idx = np.random.randint(len(X_t_train), size=n)
    choice_X = X_t_train[idx,:]
    choice_y = y_t_train[idx]
    X_s_train = np.vstack((X_s_train, choice_X))
    y_s_train = np.hstack((y_s_train, choice_y))
    return X_s_train, y_s_train

X_s_train, y_s_train = mix_data(X_s_train, y_s_train, X_t_train, y_t_train, 0)

clf = LogisticRegression().fit(X_s_train, y_s_train)
print(clf.score(X_s_val, y_s_val))
print(clf.score(X_t_test, y_t_test))

print(X_s_train.shape)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_s_train, y_s_train)
print(clf.score(X_t_test, y_t_test))

clf = tree.DecisionTreeClassifier()
clf.fit(X_s_train, y_s_train)
print(clf.score(X_t_test, y_t_test))
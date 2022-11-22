import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data = np.array(pd.read_csv('./heart_disease_health_indicators_BRFSS2015.csv', header = 0, usecols = ['HeartDiseaseorAttack','HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','Diabetes','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost', 'GenHlth', 'MentHlth','PhysHlth','DiffWalk', 'Age', 'Income']).values)

high_income = data[np.where(data[:, -1] >= 8)]  # source domain
low_income = data[np.where(data[:, -1] <= 1)]  # target domain

X_s = high_income[:, 1:]
y_s = high_income[:, 0]

X_t = low_income[:, 1:]
y_t = low_income[:, 0]

X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_s, y_s, test_size=0.3)
X_s_train, X_s_val, y_s_train, y_s_val = train_test_split(X_s_train, y_s_train, test_size=0.3)

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

# clf = LogisticRegression().fit(X_s_train, y_s_train)
# print(clf.score(X_s_test, y_s_test))
# print(clf.score(X_t_test, y_t_test))

# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf.fit(X_s_train, y_s_train)
# print(clf.score(X_t_test, y_t_test))
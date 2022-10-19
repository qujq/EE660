import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor

train_data = np.array(pd.read_csv('./Midterm/Pr1_Bike_train.csv', header = 0).values)
val_data = np.array(pd.read_csv('./Midterm/Pr1_Bike_val.csv', header = 0).values)
test_data = np.array(pd.read_csv('./Midterm/Pr1_Bike_test.csv', header = 0).values)

X_train = train_data[:, :train_data.shape[1] - 1]
y_train = train_data[:, train_data.shape[1] - 1]

X_val = val_data[:, :val_data.shape[1] - 1]
y_val = val_data[:, val_data.shape[1] - 1]

X_test = test_data[:, :test_data.shape[1] - 1]
y_test = test_data[:, test_data.shape[1] - 1]

y_train_mean = np.mean(y_train)

y_train = [y / y_train_mean for y in y_train]
y_val = [y / y_train_mean for y in y_val]
y_test = [y / y_train_mean for y in y_test]

# no regularization
# reg_no_regularization = LinearRegression(fit_intercept=False).fit(X_train, y_train)
# y_pred_no_regularization = reg_no_regularization.predict(X_val)
# mse_no_regularization = mean_squared_error(y_pred_no_regularization, y_val)
# print(mse_no_regularization)

# for log_lambda in range(-10, 11, 1):
#     cur_lambda = 2 ** log_lambda

#     reg_l1 = linear_model.Lasso(alpha=cur_lambda, fit_intercept=False)
#     reg_l1.fit(X_train, y_train)
#     y_pred_l1 = reg_l1.predict(X_val)
#     mse_l1 = mean_squared_error(y_pred_l1, y_val)

#     reg_l2 = linear_model.Ridge(alpha=cur_lambda, fit_intercept=False)
#     reg_l2.fit(X_train, y_train)
#     y_pred_l2 = reg_l2.predict(X_val)
#     mse_l2 = mean_squared_error(y_pred_l2, y_val)
#     print("lambda: ", cur_lambda,  " mse (l1): ", mse_l1, " mse (l2): ", mse_l2)

min_mse = 10
res_thod = 0
for i in range(1, 10000, 100):
    # reg_RANSAC = RANSACRegressor(random_state=0, min_samples=0.98, max_trials=100).fit(X_train, y_train)
    reg_RANSAC = RANSACRegressor(random_state=0, min_samples=0.98, max_trials=100, residual_threshold=0.23).fit(X_train, y_train)
    y_pred_RANSAC = reg_RANSAC.predict(X_val)
    mse_RANSAC = mean_squared_error(y_pred_RANSAC, y_val)
    if mse_RANSAC < min_mse:
        min_mse = mse_RANSAC
        res_thod = i
    print(i, " : ", mse_RANSAC)
print(min_mse, res_thod)

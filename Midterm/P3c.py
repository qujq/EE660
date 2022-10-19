import numpy as np

np.random.seed(2022)

def generate_poly(degree):
  np.random.seed(2022)
  return np.random.rand(degree + 1) - 0.5

f_x_coeff = generate_poly(40)
f_x_coeff = np.flip(f_x_coeff)

def get_fx(f_x_coeff, x):
  res = 0
  for i in range(len(f_x_coeff)):
    res += f_x_coeff[i] * (x ** i)
  return res

k = 3
N = 10
E_D_E_in_list = []
E_D_E_out_list = []
for _ in range(1000):
  X_list = [np.random.uniform(-1, 1.0000001) for _ in range(N)]

  X_input = []
  for X_i in X_list:
    X_i_pow_list = []
    for i in range(k + 1):
      X_i_pow_list.append(X_i ** i)
    X_input.append(X_i_pow_list)

  Y = [get_fx(f_x_coeff, x) for x in X_list]

  import numpy as np
  from sklearn.linear_model import LinearRegression
  reg = LinearRegression(fit_intercept=False).fit(X_input, Y)
  h_k_coeff = reg.coef_

  # E in

  E_in_each_list = []
  h_k_x_pred = reg.predict(X_input)

  for i in range(len(Y)):
    E_in_each_list.append((h_k_x_pred[i] - Y[i]) ** 2)
  E_in = np.mean(E_in_each_list)
  E_D_E_in_list.append(E_in)

  # E out

  X_out = np.linspace(-1, 1, 10000)
  E_out_each_list = []
  for x_i in X_out:
    E_out_each_list.append((get_fx(h_k_coeff, x_i) - get_fx(f_x_coeff, x_i) ** 2))
  E_out = np.mean(E_out_each_list)
  E_D_E_out_list.append(E_out)

print(np.mean(E_D_E_in_list))
print(E_D_E_in_list)
print(np.mean(E_D_E_out_list))
print(E_D_E_out_list)
import numpy as np

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

k = 5
N = 30

X_list = [np.random.uniform(-1, 1.0000001) for _ in range(N)]
print(X_list)
print(len(X_list))

X_input = []
for X_i in X_list:
  X_i_pow_list = []
  for i in range(k + 1):
    X_i_pow_list.append(X_i ** i)
  X_input.append(X_i_pow_list)

print(X_input)
print(len(X_input))
print(len(X_input[0]))
Y = [get_fx(f_x_coeff, x) for x in X_list]

import numpy as np
from sklearn.linear_model import LinearRegression
reg = LinearRegression(fit_intercept=False).fit(X_input, Y)
print(reg.coef_)
print(reg.intercept_)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
x = [i * 0.01 for i in range(-100, 101)]
y = [get_fx(f_x_coeff, x_i) for x_i in x]
h = [get_fx(reg.coef_, x_i) for x_i in x]
plt.plot(x, y, "r", label='f(x)')
plt.plot(x, h, "g", label='h_k(x)')
plt.scatter(X_list, Y, label='data points')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
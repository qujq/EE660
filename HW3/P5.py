import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
  # return math.exp(3 * x) / (1 + math.exp(3 * x))
  return math.exp(3 * x) / (1 + math.exp(3 * x)) + np.random.normal(0.1, 0.004, 1)

def axb(a, x, b):
  return np.multiply(a, x) + b

plt.figure()

x = [i * 0.01 for i in range(-100, 101)]
y = [f(i) for i in x]

a_list = []
b_list = []

x1_list = []
x2_list = []

a_best = 0
b_best = 0
min_mse = 9999999

for _ in range(100):
  x1 = np.random.uniform(0, 1.0000001)
  y1 = f(x1)
  x2 = np.random.uniform(0, 1.0000001)
  y2 = f(x2)
  x1_list.append(x1)
  x2_list.append(x2)

  a = (y1 - y2) / (x1 - x2)
  b = (x1 * y2 - x2 * y1) / (x1 - x2)
  a_list.append(a)
  b_list.append(b)

  plt.plot(x, axb(a, x, b), 'lightgrey')

  mse_list = []
  for _ in range(20):
    x_test = np.random.uniform(0, 1.0000001)
    mse = (a * x_test + b - f(x_test)) ** 2
    mse_list.append(mse)
  if np.mean(mse_list) < min_mse:
    a_best = a
    b_best = b
    min_mse = np.mean(mse_list)

plt.plot(x, y, 'r')
hg_mean = [np.mean(a_list) * i + np.mean(b_list) for i in x]
plt.plot(x, hg_mean, 'b')
plt.show()

a_mean = np.mean(a_list)
b_mean = np.mean(b_list)
print("a: ", a_mean)
print("b: ", b_mean)


bias_list = []
var_list = []
EdEout_list = []

for i in range(len(a_list)):
  x1 = np.random.uniform(0, 1.0000001)
  x2 = np.random.uniform(0, 1.0000001)

  hg_mean_x_fx = (a_mean * x1 + b_mean - f(x1)) ** 2
  bias_list.append(hg_mean_x_fx)
  hg_mean_x_fx = (a_mean * x2 + b_mean - f(x2)) ** 2
  bias_list.append(hg_mean_x_fx)

  a = a_list[i]
  b = b_list[i]

  hg_d_hg_mean = ((a * x1 + b - (a_mean * x1 + b_mean)) ** 2 + (a * x2 + b - (a_mean * x2 + b_mean)) ** 2) / 2
  var_list.append(hg_d_hg_mean)

  hg_d_fx = ((a * x1 + b - f(x1)) ** 2 + (a * x2 + b - f(x2)) ** 2) / 2
  EdEout_list.append(hg_d_fx)

print("bias:", np.mean(bias_list))
print("var:", np.mean(var_list))
print("EdEout(hg):", np.mean(EdEout_list))

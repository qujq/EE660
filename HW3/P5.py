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

pred_list = []
x1_list = []
x2_list = []

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

plt.plot(x, y, 'r')
hg_mean = [np.mean(a_list) * i + np.mean(b_list) for i in x]
plt.plot(x, hg_mean, 'b')
plt.show()
print("a: ", np.mean(a_list))
print("b: ", np.mean(b_list))


bias_list = [(hg_mean[i] - y[i]) ** 2 for i in range(len(hg_mean))]
var_list = []

for i in range(len(x1_list)):
  hg1 = ((a_list[i] * x1_list[i] + b_list[i]) - (np.mean(a_list) * x1_list[i] + np.mean(b_list))) ** 2
  hg2 = ((a_list[i] * x2_list[i] + b_list[i]) - (np.mean(a_list) * x2_list[i] + np.mean(b_list))) ** 2
  Ed = (hg1 + hg2) / 2
  var_list.append(Ed)

print("bias:", np.mean(bias_list))
print("var:", np.mean(var_list))
print("EdEout(hg):", np.mean(bias_list) + np.mean(var_list))

import matplotlib.pyplot as plt
import numpy as np

def probability(eps, N, mu, M):
  num_trial = 1000
  k = np.random.binomial(N, mu, (num_trial, M))
  p = np.abs(k / N - mu).max(axis = 1) > eps
  count = 0
  for res in p:
    if res:
      count += 1
  return count / num_trial

def hoeffiding(N, eps):
  return 2 * np.exp(-2 * N * eps * eps)

P_6 = []
P_60 = []
hoeffding_list_6 = []
hoeffding_list_60 = []
mu = 0.5
M = 2
eps_range = []
for i in range(1, 101, 1):
  eps = i / 100
  eps_range.append(eps)
  P_6.append(probability(eps, 6, mu, M))
  hoeffding_list_6.append(hoeffiding(6, eps))
  P_60.append(probability(eps, 60, mu, M))
  hoeffding_list_60.append(hoeffiding(60, eps))

plt.figure()
plt.xlabel("epsilon")
plt.plot(eps_range, P_6, "r")
plt.plot(eps_range, hoeffding_list_6, "c")

plt.plot(eps_range, P_60, "g")
plt.plot(eps_range, hoeffding_list_60, "b")
plt.legend(["N = 6", "hoeffiding with N = 6", "N = 60", "hoeffiding with N = 60"])
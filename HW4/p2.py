
import numpy as np
import math
Nt = 1000
Ns = 100000
N = Ns + Nt
beta = Nt / N

for alpha in [0.1, 0.5, 0.9]:
  eab = 2 * (1 - alpha) * (0.5 * 0.1) \
  + 4 * math.sqrt(alpha ** 2 / beta + (1 - alpha) ** 2 / (1 - beta)) \
  * math.sqrt(2 / N * 10 * np.log(2 * (N + 1)) + 2 / N * np.log(8 / 0.1))

  print(eab)

import numpy as np
import math
import matplotlib.pyplot as plt

def get_eab(alpha, beta, N):
  eab = 2 * (1 - alpha) * (0.5 * 0.1) \
  + 4 * math.sqrt(alpha ** 2 / beta + (1 - alpha) ** 2 / (1 - beta)) \
  * math.sqrt(2 / N * 10 * np.log(2 * (N + 1)) + 2 / N * np.log(8 / 0.1))
  return eab 

plt.figure(figsize=(8, 6), dpi=80)
Ns = 1000
for Nt in [10,100,1000,10000]:
  N = Ns + Nt
  beta = Nt / N
  alpha = np.linspace(0, 1, 201)
  eab = [get_eab(a, beta, N) for a in alpha]
  plt.plot(alpha, eab, label="Nt =" + str(Nt) + " beta = " + str(beta))
plt.legend()
plt.xlabel("alpha")
plt.ylabel("epsilon")
plt.show()

import numpy as np
import math
import matplotlib.pyplot as plt

def get_eab(alpha, beta, N):
  eab = 2 * (1 - alpha) * (0.5 * 0.1) \
  + 4 * math.sqrt(alpha ** 2 / beta + (1 - alpha) ** 2 / (1 - beta)) \
  * math.sqrt(2 / N * 10 * np.log(2 * (N + 1)) + 2 / N * np.log(8 / 0.1))
  return eab 

plt.figure(figsize=(8, 6), dpi=80)
Nt = 100
for Ns in [10,100,1000,10000]:
  N = Ns + Nt
  beta = Nt / N
  alpha = np.linspace(0, 1, 201)
  eab = [get_eab(a, beta, N) for a in alpha]
  plt.plot(alpha, eab, label="Ns =" + str(Ns) + " beta = " + str(beta))
plt.legend()
plt.xlabel("alpha")
plt.ylabel("epsilon")
plt.show()

import numpy as np
import math
import matplotlib.pyplot as plt

def get_eab(alpha, beta, N):
  eab = 2 * (1 - alpha) * (0.5 * 0.1) \
  + 4 * math.sqrt(alpha ** 2 / beta + (1 - alpha) ** 2 / (1 - beta)) \
  * math.sqrt(2 / N * 10 * np.log(2 * (N + 1)) + 2 / N * np.log(8 / 0.1))
  return eab 

plt.figure(figsize=(8, 6), dpi=80)

for beta in [0.01,0.1,0.5]:
  alpha = 0.5
  N = np.linspace(1000, 100000, 201)
  eab = [get_eab(alpha, beta, n) for n in N]
  plt.plot(N, eab, label="alpha =" + str(alpha) + " beta = " + str(beta))
plt.legend()
plt.xlabel("N")
plt.ylabel("epsilon")
plt.show()

import numpy as np
import math
import matplotlib.pyplot as plt

def get_eab(alpha, beta, N):
  eab = 2 * (1 - alpha) * (0.5 * 0.1) \
  + 4 * math.sqrt(alpha ** 2 / beta + (1 - alpha) ** 2 / (1 - beta)) \
  * math.sqrt(2 / N * 10 * np.log(2 * (N + 1)) + 2 / N * np.log(8 / 0.1))
  return eab 

plt.figure(figsize=(8, 6), dpi=80)

for beta in [0.01,0.1,0.5]:
  alpha = beta
  N = np.linspace(1000, 100000, 201)
  eab = [get_eab(alpha, beta, n) for n in N]
  plt.plot(N, eab, label="alpha =" + str(alpha) + " beta = " + str(beta))
plt.legend()
plt.xlabel("N")
plt.ylabel("epsilon")
plt.show()
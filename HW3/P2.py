from numpy.core import multiarray
import numpy as np

data_points = []

mu = 0.5

for _ in range(int(mu * 1000)):
  data_points.append(0)

for _ in range(int(1000 - mu * 1000)):
  data_points.append(1)

np.random.shuffle(data_points)

N = 10


def random_draw(data_points, N):
  res = []
  error_count = 0
  for _ in range(N):
    cur_draw = np.random.choice(data_points, 1)[0]
    if cur_draw == 0:
      error_count += 1
    res.append(cur_draw)
  error_rate = error_count / N
  return error_rate

# random_draw(data_points, N)

# b

num_choose = 100

error_rate_list_100 = []

for _ in range(num_choose):
  error_rate_list_100.append(random_draw(data_points, N))

print(np.max(error_rate_list_100))
print(np.min(error_rate_list_100))
print(np.mean(error_rate_list_100))
print(np.std(error_rate_list_100))

diff_count = 0
in_range_count = 0
learned_count = 0

for n in error_rate_list_100:
  if n != mu:
    diff_count += 1
  if np.abs(n - mu) < 0.05:
    in_range_count += 1
  if n <= 0.45 or n >= 0.55:
    learned_count += 1

print(diff_count)
print(in_range_count)
print(error_rate_list_100)
print(learned_count)




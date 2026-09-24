import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

x1 = rng.uniform(low=-0.5, high=0.5, size=10000)
x2 = rng.uniform(low=-0.5, high=0.5, size=10000)

x1_plus_x2_over_2 = list(map(lambda x: (x[0] + x[1])/2, zip(x1, x2)))

plt.hist(x1_plus_x2_over_2, bins=50, color='skyblue', edgecolor='black')
plt.show()

x3 = rng.uniform(low=-0.5, high=0.5, size=10000)
x4 = rng.uniform(low=-0.5, high=0.5, size=10000)

sum_x1_to_x4_over_4 = list(map(lambda x: (x[0] + x[1] + x[2] + x[3])/4, zip(x1, x2, x3, x4)))

plt.hist(sum_x1_to_x4_over_4, bins=50, color='skyblue', edgecolor='black')
plt.show()

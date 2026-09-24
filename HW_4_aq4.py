import numpy as np
import matplotlib.pyplot as plt
import math
import random

x = np.arange(-6, 8, 0.001)

WeightedUniformDistribution = tuple[float, float, float]

mixands: list[WeightedUniformDistribution] = [
    (0.1859, -4, -2),
    (0.0961, -2, -1),
    (0.1055, -0.75, 0),
    (0.2104, 0, 1),
    (0.0678, 3, 5.5),
    (0.1950, 4, 6),
    (0.1393, -0.9, 7),
]



def p_at_x(x) -> float:
    sum = 0
    for mixand in mixands:
        if x < mixand[1] or x >= mixand[2]:
            continue
        b_minus_a = mixand[2] - mixand[1]
        sum += mixand[0] / b_minus_a
    return sum


p_x = [p_at_x(val) for val in x]
maximum = 0
for p in p_x:
    maximum = max(maximum, p)

print(f"Maxiumim p(x) = {maximum}")

plt.plot(x, p_x)
plt.show()

# Calculate the mean from the graph
# Mean = sum (x * delta_x * p(x))

estimated_mean = 0
for x_value, p_x_value in zip(x, p_x):
    estimated_mean += x_value * p_x_value * 0.001

print(f"Estimated Mean: {estimated_mean}")

actual_mean = 0
for m in mixands:
    actual_mean += ((m[1] + m[2]) / 2) * m[0]

print(f"Calculated Mean: {actual_mean}")

estimated_variance = 0
for x_value, p_x_value in zip(x, p_x):
    estimated_variance += x_value * x_value * p_x_value * 0.001

estimated_variance -= estimated_mean * estimated_mean

print (f"Estimated variance = {estimated_variance}")

differential_entropy = 0
for p_x_value in p_x:
    if p_x_value == 0:
        continue
    differential_entropy += -p_x_value * math.log(p_x_value) * 0.001

print(f"DE: {differential_entropy}")

kl_divergence = 0
for p_x_value in p_x:
    if p_x_value == 0:
        continue
    kl_divergence += p_x_value *math.log(p_x_value / (1/11)) * 0.001

print(f"kl_divergence: {kl_divergence}")

# Use rejection sampling.
# First find maximum value of p(x)
maximum = 0
for p in p_x:
    maximum = max(maximum, p)

print(f"Maxiumim p(x) = {maximum}")

# P(x) is non-zero on the range [-4, 7]
sample_set_100 = []
while len(sample_set_100) <100:
    x = random.uniform(-4, 7)
    p_x = p_at_x(x)
    # Now scale to 'maximum'
    threshold = random.uniform(0, maximum)
    if p_x >= threshold:
        sample_set_100.append(x)

sample_set_1000 = []
while len(sample_set_1000) <1000:
    x = random.uniform(-4, 7)
    p_x = p_at_x(x)
    # Now scale to 'maximum'
    threshold = random.uniform(0, maximum)
    if p_x >= threshold:
        sample_set_1000.append(x)

sample_set_50000 = []
while len(sample_set_50000) <50000:
    x = random.uniform(-4, 7)
    p_x = p_at_x(x)
    # Now scale to 'maximum'
    threshold = random.uniform(0, maximum)
    if p_x >= threshold:
        sample_set_50000.append(x)

plt.hist(sample_set_100, bins=1000)
plt.show()

plt.hist(sample_set_1000, bins=1000)
plt.show()

plt.hist(sample_set_50000, bins=1000)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from functools import reduce

# Create our list
delta_x = 0.001
x = np.arange(-6, 8, delta_x)

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

# Here is where we plot the graph for question part b)
plt.xlabel(r"$x$")
plt.ylabel(r"$p(x)$")
plt.title(r"Graph of $p(x)$ vs $x$ for $x\in[-6, 8]$")
plt.plot(x, p_x, linewidth=1.0)
plt.show()

# The following code is for quesiton part c)
print("Question part c).")
estimated_mean = 0
for x_value, p_x_value in zip(x, p_x):
    estimated_mean += x_value * p_x_value * delta_x

print(f"Estimated Mean: {estimated_mean:.6f}")

calculated_mean = 0
for m in mixands:
    calculated_mean += ((m[1] + m[2]) / 2) * m[0]

print(f"Calculated Mean: {calculated_mean:.6f}")

estimated_variance = 0
for x_value, p_x_value in zip(x, p_x):
    estimated_variance += x_value * x_value * p_x_value * delta_x

estimated_variance -= estimated_mean * estimated_mean

print(f"Estimated variance = {estimated_variance:.6f}")

calculated_variance = 0
for m in mixands:
    calculated_variance += (m[0] / (3 * (m[2] - m[1]))) * (m[2] ** 3 - m[1] ** 3)
calculated_variance -= calculated_mean**2
print(f"Calculated variance = {calculated_variance:.6f}")


differential_entropy = 0
for p_x_value in p_x:
    if p_x_value == 0:
        continue
    differential_entropy += -p_x_value * math.log(p_x_value) * delta_x

print(f"Differential Entropy: {differential_entropy:.6f}")

kl_divergence = 0
for p_x_value in p_x:
    if p_x_value == 0:
        continue
    kl_divergence += p_x_value * math.log(p_x_value / (1 / 11)) * delta_x

print(f"Kullback-Leibler Divergence: {kl_divergence:.6f}")

# Use rejection sampling.
# First find maximum value of p(x)
maximum = 0
for p in p_x:
    maximum = max(maximum, p)


# The following code is for quesiton part d)
def generate_sample_set(n: int) -> list[float]:
    sample_set = list[float]()
    while len(sample_set) < n:
        # P(x) is non-zero on the range [-4, 7]
        x = random.uniform(-4, 7)
        p_x = p_at_x(x)
        # Now scale to 'maximum'
        threshold = random.uniform(0, maximum)
        if p_x >= threshold:
            sample_set.append(x)
    return sample_set


sample_set_100 = generate_sample_set(100)
sample_set_1000 = generate_sample_set(1000)
sample_set_50000 = generate_sample_set(50000)

plt.xlabel(r"$x$")
plt.ylabel(r"# samples at $x$")
plt.title(r"Graph of frequencies of 100 random samples drawn from $p(x)$")
plt.hist(sample_set_100, bins=1000)
plt.show()

plt.xlabel(r"$x$")
plt.ylabel(r"# samples at $x$")
plt.title(r"Graph of frequencies of 1000 random samples drawn from $p(x)$")
plt.hist(sample_set_1000, bins=1000)
plt.show()

plt.xlabel(r"$x$")
plt.ylabel(r"# samples at $x$")
plt.title(r"Graph of frequencies of 50000 random samples drawn from $p(x)$")
plt.hist(sample_set_50000, bins=1000)
plt.show()

# The following code is for quesiton part e)
print("\n\nQuestion part e).")
# Monte Carlo approximations
# Expected value of X.
e_x_100_samples = reduce(lambda acc, x: acc + x, sample_set_100) / len(sample_set_100)
e_x_1000_samples = reduce(lambda acc, x: acc + x, sample_set_1000) / len(
    sample_set_1000
)
e_x_50000_samples = reduce(lambda acc, x: acc + x, sample_set_50000) / len(
    sample_set_50000
)
print(
    f"M-C approximation of E[x] (100-sample, 1000-sample, 50000-sample)"
    f" = ({e_x_100_samples:.6f}, {e_x_1000_samples:.6f}, {e_x_50000_samples:.6f})"
)

var_x_100_samples = reduce(
    lambda acc, x: acc + (x - e_x_100_samples) ** 2, sample_set_100
) / len(sample_set_100)
var_x_1000_samples = reduce(
    lambda acc, x: acc + (x - e_x_1000_samples) ** 2, sample_set_1000
) / len(sample_set_1000)
var_x_50000_samples = reduce(
    lambda acc, x: acc + (x - e_x_50000_samples) ** 2, sample_set_50000
) / len(sample_set_50000)
print(
    f"M-C approximation of var(x) (100-sample, 1000-sample, 50000-sample)"
    f" = ({var_x_100_samples:.6f}, {var_x_1000_samples:.6f}, {var_x_50000_samples:.6f})"
)

de_px_100_samples = reduce(
    lambda acc, x: acc + (-math.log(p_at_x(x)) if p_at_x(x) != 0 else 0), sample_set_100
) / len(sample_set_100)
de_px_1000_samples = reduce(
    lambda acc, x: acc + (-math.log(p_at_x(x)) if p_at_x(x) != 0 else 0),
    sample_set_1000,
) / len(sample_set_1000)
de_px_50000_samples = reduce(
    lambda acc, x: acc + (-math.log(p_at_x(x)) if p_at_x(x) != 0 else 0),
    sample_set_50000,
) / len(sample_set_50000)

print(
    f"M-C approximation of differential entropy (100-sample, 1000-sample, 50000-sample)"
    f" = ({de_px_100_samples:.6f}, {de_px_1000_samples:.6f}, {de_px_50000_samples:.6f})"
)

kl_px_100_samples = reduce(
    lambda acc, x: acc + (math.log(p_at_x(x) / (1 / 11)) if p_at_x(x) != 0 else 0),
    sample_set_100,
) / len(sample_set_100)
kl_px_1000_samples = reduce(
    lambda acc, x: acc + (math.log(p_at_x(x) / (1 / 11)) if p_at_x(x) != 0 else 0),
    sample_set_1000,
) / len(sample_set_1000)
kl_px_50000_samples = reduce(
    lambda acc, x: acc + (math.log(p_at_x(x) / (1 / 11)) if p_at_x(x) != 0 else 0),
    sample_set_50000,
) / len(sample_set_50000)

print(
    f"M-C approximation of K-L divergence (100-sample, 1000-sample, 50000-sample)"
    f" = ({kl_px_100_samples:.6f}, {kl_px_1000_samples:.6f}, {kl_px_50000_samples:.6f})"
)

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

k = 398600
r_0 = 6678

Omega_0_squared = k/(r_0^3)
Omega_0 = np.sqrt(Omega_0_squared)

print(f"Omega_0 = {Omega_0}")

A = [[0, 1, 0, 0], [3 * Omega_0_squared, 0, 0, 2]]

print(f"A = {A}")
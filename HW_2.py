import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

k = 398600
r_0 = 6678

Omega_0 = np.sqrt(k/(r_0^3))

print(f"Omega_0 = {Omega_0}")
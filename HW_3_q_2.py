import numpy as np
import matplotlib.pyplot as plt

# Put the original Z-data into matricies
z_0 = np.array([[100, 20]]).T
z_1 = np.array([[43.6658, 39.2815]]).T
z_2 = np.array([[40.5785, 40.3382]]).T
z_3 = np.array([[40.4093, 40.3961]]).T
z_4 = np.array([[40.4000, 40.3993]]).T
z_5 = np.array([[40.3995, 40.3995]]).T

# Now calculate the y values...
y_1 = z_1 - z_0
y_2 = z_2 - z_1
y_3 = z_3 - z_2
y_4 = z_4 - z_3
y_5 = z_5 - z_4

# Build the big Y matrix
Y = np.concatenate((y_1, y_2, y_3, y_4, y_5), axis=0)

# Build the RHS...
i_2_2 = np.identity(2)
M = np.concatenate(
    (
        (z_0[0] - z_0[1]) * i_2_2,
        (z_1[0] - z_1[1]) * i_2_2,
        (z_2[0] - z_2[1]) * i_2_2,
        (z_3[0] - z_3[1]) * i_2_2,
        (z_4[0] - z_4[1]) * i_2_2,
    ),
    axis=0,
)

M_t_M = M.T @ M
M_t_M_inv = np.linalg.inv(M_t_M)
x = M_t_M_inv @ M.T @ Y
print(f"[Lambda, Mu] = {x.T[0]}")


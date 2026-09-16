from scipy.io import loadmat
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt


# From CT solution, import A and B, matricies
A = np.array([[0, 1, 0, 0], [-2, 0, 1, 0], [0, 0, 0, 1], [1, 0, -2, 0]])
B = np.array([[0, 0], [-1, 0], [0, 0], [1, 1]])

# Construct "A-hat" matrix
A_B = np.concatenate((A, B), axis=1)
A_hat = np.concatenate((A_B, np.zeros((2, 6))), axis=0)

delta_t = 0.05
# Now calulate the e^(A_hat * delta_t)
a_hat_delta_t = A_hat * delta_t
e_A_hat_delta_t = expm(a_hat_delta_t)

# expm(a_hat_delta_t) gives the F and G matricies of the DT LTI SS model
# expm(a_hat_delta_t) = |F, G|
#                       |0, 0|
# Pick of the parts and output the
F = e_A_hat_delta_t[0:4, 0:4]
G = e_A_hat_delta_t[0:4, 4:6]
print("Question 1.(a)")
print("DT LTI representation is in the form x(k+1) = Fx(k) + Gu(k)")
print("Where F and G matricies are given by...\n")
print(f"F =\n{F}\n")
print(f"G =\n{G}\n")

print("Nyquist limit can be found by looking at eigen-values of the original A matrix")
print("Calculate eigen-values with numpy.linalg.eigvals and find the eigenvalue with the largest modulus...")
A_eigenvals = np.linalg.eigvals(A)
largest_modulus_eigenvalue = max(map(lambda e_val: np.abs(e_val), A_eigenvals))
print(f"Largest magitude eigenvalue (Lambda_max) has modulus {largest_modulus_eigenvalue}.")
print(f"Nyquist limit is given by pi/Lambda_max = {np.pi/largest_modulus_eigenvalue}. This is")
print("larger than the 0.05s sample rate given in the question, therefore no risk of aliasing.\n\n")

print("Question 1.(b)")
print("DT system is observable iff LTI Observability Matrix O, given by")
print(" O =   |H   |")
print("       |HF  |")
print("       |HF^2|")
print("       |HF^3|")
print(
    "is full-column rank. To confirm this, look at rank of Gram Matrix (O-transpose * O)"
)
# If there's an easier way to build this please let me know!!
# From problem spec, we are given H.
H = np.array([[1, 0, 0, 0], [0, 1, 0, -1]])
HF = np.matmul(H, F)
HFF = np.matmul(HF, F)
HFFF = np.matmul(HFF, F)
O = np.append(np.append(np.append(H, HF, axis=0), HFF, axis=0), HFFF, axis=0)
O_transpose_O = np.matmul(O.T, O)
print(f"Rank of Gram Matrix (and therefore rank of O) is {np.linalg.matrix_rank(O_transpose_O)}, which is the same as 'n', therefore system is observable.\n\n")

print("Question 1.(c)")
print("See hand-written derivation for this question.\n\n")

print("Question 1.(d)")
print("Since data provided in log file goes from y(1) to y(100), first calculate x(1)...")
data = loadmat("hw3problem1data.mat")
# See written explanation of this derivation.
# "Udata" goes from u(0) to u(100)
# "Ydata" goes from y(1) to y(100)
u_original = np.array(data["Udata"])
y_original = np.array(data["Ydata"])

# Flaten Y into a 200*1 matrix.
y = y_original.reshape(-1, 1)
# Flatten U into 202*1 matrix. 
u = u_original.reshape(-1, 1)
print(f"len(y) = {len(y)}, len(u) = {len(u)}")

big_daddy = np.zeros((200, 200))
for i in range (0, 100):
    element = H @ np.linalg.matrix_power(F, i) @ G
    for offset in range (0, 100-i):
        big_daddy[(i+offset)*2: (i+offset+1)*2, i*2: (i+1)*2] = element

LHS = y-(big_daddy @ u[0:200])

# Now construct the RHS matrix
RHS = np.zeros((200, 4))
for i in range(0, 100):
    element = H @ np.linalg.matrix_power(F, i+1)
    RHS[i*2: (i+1)*2, 0:5] = element

RHS_T_RHS = np.matmul(RHS.T, RHS)

print(RHS_T_RHS)
RHS_T_RHS_INV = np.linalg.inv(RHS_T_RHS)

x0 = RHS_T_RHS_INV @ RHS.T @ LHS 

print(f"x0 = {x0}")
# y1 = np.array([y[0]]).T
# y2 = np.array([y[1]]).T
# y3 = np.array([y[2]]).T
# y4 = np.array([y[3]]).T

# u1 = np.array([u[1, :]]).T
# u2 = np.array([u[2, :]]).T
# u3 = np.array([u[3, :]]).T
# u4 = np.array([u[4, :]]).T

# Gu1 = np.matmul(G, u1)
# Gu2 = np.matmul(G, u2)
# Gu3 = np.matmul(G, u3)
# Gu4 = np.matmul(G, u4)
# FGu1 = np.matmul(F, Gu1)
# FGu2 = np.matmul(F, Gu2)
# FFGu1 = np.matmul(F, FGu1)
# # Build the Y matrix...
# Y = np.concatenate((y1, y2 - np.matmul(H, Gu1), y3- np.matmul(H, Gu2 - FGu1), y4 - np.matmul(H, Gu3- FGu2- FFGu1)), axis=0)
# O_transpose_O_inverse = np.linalg.inv(O_transpose_O)

# x1 = np.matmul(np.matmul(O_transpose_O_inverse, O.T), Y)
# print("System state at k=1 is x(k=1):")
# print(x1)
# print("\nNow calculate x(k=0) using x(0) = F_inv * (x(1) - Gu(0)):")
# u0 = np.array([u[0, :]]).T
# Gu0 = np.matmul(G, u0)
# F_inverse = np.linalg.inv(F)
# x0 = np.matmul(F_inverse, x1 - Gu0)
# print("x(k=0) =")
# print(x0)

# Now generate predictions for the rest of the system states from t = 0 to t = 5.
X_states = [x0]
Y_predictions = []
timestamps = []
previous_x = x0
for k in range(0, 100):
    # Timestamp lags, i.e. the zoh value for u is the value of u(t) at k, not k+1
    current_timestamp = 0.05 * k
    zoh_u = np.array([[np.sin(current_timestamp)], [0.1 * np.cos(current_timestamp)]])
    x_k_plus_1 = (F @ previous_x) + (G @ zoh_u)
    y_k_plus_1 = (H @ x_k_plus_1)
    X_states.append(x_k_plus_1)
    Y_predictions.append(y_k_plus_1)
    timestamps.append(0.05 * (k+1))
    previous_x = x_k_plus_1

Y_1 = []
Y_2 = []
for y_prediction in Y_predictions:
    Y_1.append((y_prediction.T)[0, 0])
    Y_2.append((y_prediction.T)[0, 1])

YY_1 = []
YY_2 = []
for y_val in y_original:
    YY_1.append(y_val[0])
    YY_2.append(y_val[1])

# print(f"len(Y_1) = {len(Y_1)}, len(YY_1) = {len(YY_1)}")
plt.plot(timestamps, Y_1, label="Y1", color='red')
plt.plot(timestamps, Y_2, label="Y2", color='green')
plt.plot(timestamps, YY_1, label="YY1", color='red')
plt.plot(timestamps, YY_2, label="YY2", color='green')
plt.title("State Vector Components vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Pertubation (rad/s)")
plt.legend()
plt.show()



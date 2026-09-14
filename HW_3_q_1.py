from scipy.io import loadmat
from scipy.linalg import expm
import numpy as np

# From CT solution, import A and B, matricies
A = np.array([[0, 1, 0, 0], [-2, 0, 1, 0], [0, 0, 0, 1], [1, 0, -2, 0]])
B = np.array([[0, 0], [-1, 0], [0, 0], [1, 1]])

# Construct "A-hat" matrix
A_B = np.append(A, B, axis=1)
A_hat = np.append(A_B, np.zeros((2, 6)), axis=0)

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
O_transpose_O = np.matmul(np.transpose(O), O)
print(f"Rank of Gram Matrix (and therefore rank of O) is {np.linalg.matrix_rank(O_transpose_O)}, which is the same as 'n', therefore system is observable.\n\n")

print("Question 1.(c)")
print("Since data provided in log file goes from y(1) to y(100), first calculate x(1)...")
data = loadmat("hw3problem1data.mat")
# "Udata" goes from u(0) to u(100)
# "Ydata" goes from y(1) to y(100)
u = np.array(data["Udata"])
y = np.array(data["Ydata"])

print(y)
y1 = np.array(y[0])
print("********")
print(y1)
print(y1.T)
y2 = np.transpose(y[1])
y3 = np.transpose(y[2])
y4 = np.transpose(y[3])

u1 = np.transpose(u[1, :])
u2 = np.transpose(u[2, :])
u3 = np.transpose(u[3, :])
u4 = np.transpose(u[4, :])

# Build the Y matrix...
Y = np.append(y1, y2 - np.matmul(H, np.matmul(G, u1)), axis = 0)
print(Y)

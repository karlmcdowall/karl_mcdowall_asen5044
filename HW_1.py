import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

A = np.array([[0, 0, 0],
             [0, 0, -40./3],
             [0, 5, 0]])

timestamps = np.linspace(0, 5, 200)

# Xo is our initial condition
Xo = np.array([0, 0.1, 0])
# X is a list of state vector values.
X = []
for t in timestamps:
    x = np.matmul(expm(A * t), Xo)
    X.append(x)

# Break up the state vector into elements for display
X_p = []
X_q = []
X_r = []
for x in X:
    X_p.append(x[0])
    X_q.append(x[1])
    X_r.append(x[2])

plt.plot(timestamps, X_p, label=r"$\Delta p$", color='red')
plt.plot(timestamps, X_q, label=r"$\Delta q$", color='green')
plt.plot(timestamps, X_r, label=r"$\Delta r$", color='blue')
plt.title("State Vector Components vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Pertubation (rad/s)")
plt.legend()
plt.show()

Xcalc = []
lam = 10. * np.sqrt(2/3)
for time in timestamps:
    E_to_At = np.array([[1, 0, 0],
                      [0, np.cos(time * lam), -2. * np.sqrt(2/3) * np.sin(time * lam)],
                      [0, np.sqrt(3/8) * np.sin(time * lam), np.cos(time * lam)]])
    Xcalc.append(np.matmul(E_to_At, Xo))

X_p = []
X_q = []
X_r = []
for x in X:
    X_p.append(x[0])
    X_q.append(x[1])
    X_r.append(x[2])

plt.plot(timestamps, X_p, label=r"$\Delta p$", color='red')
plt.plot(timestamps, X_q, label=r"$\Delta q$", color='green')
plt.plot(timestamps, X_r, label=r"$\Delta r$", color='blue')
plt.title("State Vector Components vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Pertubation (rad/s)")
plt.legend()
plt.show()

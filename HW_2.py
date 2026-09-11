import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def scipy_system(_t, S):
    k = 398600.0
  
    r, r_dot, theta, theta_dot = S
    dr_dt = r_dot
    dtheta_dt = theta_dot
    d2r_dt2 = (r * theta_dot * theta_dot) - k/(r**2)
    d2theta_dt2 = (-2 * theta_dot * r_dot / r)

    print(f"Theta = {theta}")
    return [dr_dt, d2r_dt2, dtheta_dt, d2theta_dt2]

def main():
    # Define constants
    k = 398600.0
    r_0 = 6678.0
    omega_0_squared = k / (r_0**3)
    omega_0 = np.sqrt(omega_0_squared)

    A = np.array(
        [
            [0, 1, 0, 0],
            [3 * omega_0_squared, 0, 0, 2 * omega_0 * r_0],
            [0, 0, 0, 1],
            [0, -2 * omega_0 / r_0, 0, 0],
        ]
    )

    delta_t = 10.0
    STM = expm(A * delta_t)

    # Initial theta and time are 0.
    theta = 0
    time = 0
    timestamps = []
    x_nom = []
    delta_x = []
    x_nom_plus_delta_x = []
    # Populate initial current_delta_x with values from question sheet.
    current_delta_x = np.array([10, -0.5, 0, 2.5e-5])
    while theta < np.pi * 2:
        current_nominal = np.array([r_0, 0, theta, omega_0])
        x_nom.append(current_nominal)
        delta_x.append(current_delta_x)
        x_nom_plus_delta_x.append(np.add(current_delta_x, current_nominal))
        timestamps.append(time)
        # Now update variables that will be used in the next iteration.
        theta += delta_t * omega_0
        time += delta_t
        current_delta_x = np.matmul(STM, current_delta_x)

    #plt.plot(timestamps, x_nom_plus_delta_x, label=r"$\Delta p$", color="red")
    # plt.plot(timestamps, X_q, label=r"$\Delta q$", color='green')
    # plt.plot(timestamps, X_r, label=r"$\Delta r$", color='blue')
    #plt.show()

    sol = solve_ivp(scipy_system, [0, 6000], [6678+10, -0.5, 0.0, omega_0 + 2.5e-5], method='RK45', t_eval=np.linspace(0, 6000, 400))
    plt.plot(sol.t, sol.y[0], label='x(t)')
    plt.show()

if __name__ == "__main__":
    main()
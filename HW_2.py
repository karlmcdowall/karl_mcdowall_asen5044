import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp


def main():
    # Define constants
    k = 398600.0
    r_0 = 6678.0
    delta_t = 10.0
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

    STM = expm(A * delta_t)

    # Initial theta and time are 0.
    theta = 0
    time = 0
    timestamps = []
    x_nom = []
    delta_x = []
    x_nom_plus_delta_x = []
    # Populate initial current_delta_x with values from question sheet.
    r_pertubation = 10.0
    r_dot_pertubation = -0.5
    theta_dot_pertubation = 2.5e-5
    current_delta_x = np.array(
        [r_pertubation, r_dot_pertubation, 0, theta_dot_pertubation]
    )
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

    delta_r = np.array([delta[0] for delta in delta_x])
    delta_r_dot = np.array([delta[1] for delta in delta_x])
    delta_theta = np.array([delta[2] for delta in delta_x])
    delta_theta_dot = np.array([delta[3] for delta in delta_x])


    plot_pertubations(timestamps, delta_r, delta_r_dot, delta_theta, delta_theta_dot)
    plot_expected_total_system_state(timestamps, x_nom_plus_delta_x)

    sol = solve_ivp(
        scipy_system,
        [timestamps[0], timestamps[len(timestamps) - 1]],
        [r_0 + r_pertubation, r_dot_pertubation, 0.0, omega_0 + theta_dot_pertubation],
        method="RK45",
        t_eval=timestamps,
    )
    assert len(timestamps) == len(sol.y[1])
    plot_rk4_total_system_state(timestamps, sol.y[0], sol.y[1], sol.y[2], sol.y[3])

    
    # plt.plot(sol.t, sol.y[3], label="x(t)")
    # plt.show()


def plot_pertubations(timestamps: list[int], delta_r: np.ndarray, delta_r_dot: np.ndarray, delta_theta: np.ndarray, delta_theta_dot: np.ndarray):
    # Make plots of pertubation-states vs time
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Time (s)")
    r_color, r_dot_color, theta_color, theta_dot_color = (
        "red",
        "blue",
        "orange",
        "green",
    )

    ax1.set_ylabel("Distance ($km$)", color=r_color)
    ax1.tick_params(axis="y", labelcolor=r_color)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines["left"].set_color(r_color)
    (line1,) = ax1.plot(
        timestamps, delta_r, label=r"$\Delta r$", color=r_color, linewidth="0.5"
    )

    ax2 = ax1.twinx()
    ax2.yaxis.tick_left()
    ax2.yaxis.set_label_position("left")
    ax2.spines["left"].set_position(("axes", -0.15))
    ax2.spines["left"].set_color(r_dot_color)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylabel(r"Speed ($km s^{-1}$)", color=r_dot_color)
    ax2.tick_params(axis="y", labelcolor=r_dot_color)
    (line2,) = ax2.plot(
        timestamps,
        delta_r_dot,
        label=r"$\Delta \dot{r}$",
        color=r_dot_color,
        linewidth="0.5",
    )

    ax3 = ax1.twinx()
    ax3.set_ylabel(r"Angle ($rad$)", color=theta_color)
    ax3.spines["right"].set_color(theta_color)
    ax3.spines[["top", "left"]].set_visible(False)
    ax3.tick_params(axis="y", labelcolor=theta_color)
    (line3,) = ax3.plot(
        timestamps,
        delta_theta,
        label=r"$\Delta\Theta$",
        color=theta_color,
        linewidth="0.5",
    )

    ax4 = ax1.twinx()
    ax4.set_ylabel(r"Angular Velocity ($rad/s$)", color=theta_dot_color)
    ax4.spines["right"].set_position(("axes", 1.15))
    ax4.spines["right"].set_color(theta_dot_color)
    ax4.spines[["top", "left"]].set_visible(False)
    ax4.tick_params(axis="y", labelcolor=theta_dot_color)
    (line4,) = ax4.plot(
        timestamps,
        delta_theta_dot,
        label=r"$\Delta \dot{\Theta}$",
        color=theta_dot_color,
        linewidth="0.5",
    )

    fig.legend(handles=[line1, line2, line3, line4])
    plt.title("Pertubation States vs. Time")
    plt.tight_layout()
    plt.show()


def plot_expected_total_system_state(
    timestamps: list[int], x_nom_plus_delta_x: list[np.ndarray]
):
    # Make plots of pertubation-states vs time
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Time (s)")
    r_color, r_dot_color, theta_color, theta_dot_color = (
        "red",
        "blue",
        "orange",
        "green",
    )

    r = [x[0] for x in x_nom_plus_delta_x]
    ax1.set_ylabel("Distance ($km$)", color=r_color)
    ax1.tick_params(axis="y", labelcolor=r_color)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines["left"].set_color(r_color)
    (line1,) = ax1.plot(timestamps, r, label=r"$r$", color=r_color, linewidth="0.5")

    ax2 = ax1.twinx()
    r_dot = [x[1] for x in x_nom_plus_delta_x]
    ax2.yaxis.tick_left()
    ax2.yaxis.set_label_position("left")
    ax2.spines["left"].set_position(("axes", -0.15))
    ax2.spines["left"].set_color(r_dot_color)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylabel(r"Speed ($km s^{-1}$)", color=r_dot_color)
    ax2.tick_params(axis="y", labelcolor=r_dot_color)
    (line2,) = ax2.plot(
        timestamps, r_dot, label=r"$\dot{r}$", color=r_dot_color, linewidth="0.5"
    )

    ax3 = ax1.twinx()
    theta = [x[2] for x in x_nom_plus_delta_x]
    ax3.set_ylabel(r"Angle ($rad$)", color=theta_color)
    ax3.spines["right"].set_color(theta_color)
    ax3.spines[["top", "left"]].set_visible(False)
    ax3.tick_params(axis="y", labelcolor=theta_color)
    (line3,) = ax3.plot(
        timestamps, theta, label=r"$\Theta$", color=theta_color, linewidth="0.5"
    )

    ax4 = ax1.twinx()
    theta_dot = [x[3] for x in x_nom_plus_delta_x]
    ax4.set_ylabel(r"Angular Velocity ($rad/s$)", color=theta_dot_color)
    ax4.spines["right"].set_position(("axes", 1.15))
    ax4.spines["right"].set_color(theta_dot_color)
    ax4.spines[["top", "left"]].set_visible(False)
    ax4.tick_params(axis="y", labelcolor=theta_dot_color)
    (line4,) = ax4.plot(
        timestamps,
        theta_dot,
        label=r"$\dot{\Theta}$",
        color=theta_dot_color,
        linewidth="0.5",
    )

    fig.legend(handles=[line1, line2, line3, line4])
    plt.title("Expected Total System States vs. Time")
    plt.tight_layout()
    plt.show()


def plot_rk4_total_system_state(
    timestamps: list[int],
    r: np.ndarray,
    r_dot: np.ndarray,
    theta: np.ndarray,
    theta_dot: np.ndarray,
):
    # Make plots of pertubation-states vs time
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Time (s)")
    r_color, r_dot_color, theta_color, theta_dot_color = (
        "red",
        "blue",
        "orange",
        "green",
    )

    ax1.set_ylabel("Distance ($km$)", color=r_color)
    ax1.tick_params(axis="y", labelcolor=r_color)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines["left"].set_color(r_color)
    (line1,) = ax1.plot(timestamps, r, label=r"$r$", color=r_color, linewidth="0.5")

    ax2 = ax1.twinx()
    ax2.yaxis.tick_left()
    ax2.yaxis.set_label_position("left")
    ax2.spines["left"].set_position(("axes", -0.15))
    ax2.spines["left"].set_color(r_dot_color)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylabel(r"Speed ($km s^{-1}$)", color=r_dot_color)
    ax2.tick_params(axis="y", labelcolor=r_dot_color)
    (line2,) = ax2.plot(
        timestamps, r_dot, label=r"$\dot{r}$", color=r_dot_color, linewidth="0.5"
    )

    ax3 = ax1.twinx()
    ax3.set_ylabel(r"Angle ($rad$)", color=theta_color)
    ax3.spines["right"].set_color(theta_color)
    ax3.spines[["top", "left"]].set_visible(False)
    ax3.tick_params(axis="y", labelcolor=theta_color)
    (line3,) = ax3.plot(
        timestamps, theta, label=r"$\Theta$", color=theta_color, linewidth="0.5"
    )

    ax4 = ax1.twinx()
    ax4.set_ylabel(r"Angular Velocity ($rad/s$)", color=theta_dot_color)
    ax4.spines["right"].set_position(("axes", 1.15))
    ax4.spines["right"].set_color(theta_dot_color)
    ax4.spines[["top", "left"]].set_visible(False)
    ax4.tick_params(axis="y", labelcolor=theta_dot_color)
    (line4,) = ax4.plot(
        timestamps,
        theta_dot,
        label=r"$\dot{\Theta}$",
        color=theta_dot_color,
        linewidth="0.5",
    )

    fig.legend(handles=[line1, line2, line3, line4])
    plt.title("Expected Total System States vs. Time")
    plt.tight_layout()
    plt.show()


def scipy_system(_t, S):
    k = 398600.0

    r, r_dot, theta, theta_dot = S
    dr_dt = r_dot
    dtheta_dt = theta_dot
    d2r_dt2 = (r * theta_dot * theta_dot) - k / (r**2)
    d2theta_dt2 = -2 * theta_dot * r_dot / r

    return [dr_dt, d2r_dt2, dtheta_dt, d2theta_dt2]


if __name__ == "__main__":
    main()

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
    plot_pertubations(
        "Linearized Model Pertubation States vs. Time",
        timestamps,
        delta_r,
        delta_r_dot,
        delta_theta,
        delta_theta_dot,
    )

    r_nom_plus_delta = np.array([x[0] for x in x_nom_plus_delta_x])
    r_dot_nom_plus_delta = np.array([x[1] for x in x_nom_plus_delta_x])
    theta_nom_plus_delta = np.array([x[2] for x in x_nom_plus_delta_x])
    theta_dot_nom_plus_delta = np.array([x[3] for x in x_nom_plus_delta_x])

    plot_total_system_state(
        "Linearized Model Expected Total System States vs. Time",
        timestamps,
        r_nom_plus_delta,
        r_dot_nom_plus_delta,
        theta_nom_plus_delta,
        theta_dot_nom_plus_delta,
    )

    # Now calculate the total system state from the original non-linear equations,
    # solving with the RK45 solver.
    sol = solve_ivp(
        scipy_system,
        [timestamps[0], timestamps[len(timestamps) - 1]],
        [r_0 + r_pertubation, r_dot_pertubation, 0.0, omega_0 + theta_dot_pertubation],
        method="RK45",
        t_eval=timestamps,
    )
    assert len(timestamps) == len(sol.y[1])

    r_nom = np.array([x[0] for x in x_nom])
    r_dot_nom = np.array([x[1] for x in x_nom])
    theta_nom = np.array([x[2] for x in x_nom])
    theta_dot_nom = np.array([x[3] for x in x_nom])
    delta_r_rk = sol.y[0] - r_nom
    delta_r_dot_rk = sol.y[1] - r_dot_nom
    delta_theta_rk = sol.y[2] - theta_nom
    delta_theta_dot_rk = sol.y[3] - theta_dot_nom
    plot_pertubations(
        "RK Pertubation from Nominal vs. Time",
        timestamps,
        delta_r_rk,
        delta_r_dot_rk,
        delta_theta_rk,
        delta_theta_dot_rk,
    )

    plot_total_system_state(
        "RK4 Itegrated Non-Linear Total System States vs. Time",
        timestamps,
        sol.y[0],
        sol.y[1],
        sol.y[2],
        sol.y[3],
    )

    plot_pertubations(
        "RK4 Pertubations minus Linearized Model Pertubations",
        timestamps,
        delta_r_rk - delta_r,
        delta_r_dot_rk - delta_r_dot,
        delta_theta_rk - delta_theta,
        delta_theta_dot_rk - delta_theta_dot,
    )

def plot_pertubations(
    title: str,
    timestamps: list[int],
    delta_r: np.ndarray,
    delta_r_dot: np.ndarray,
    delta_theta: np.ndarray,
    delta_theta_dot: np.ndarray,
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
    ax4.yaxis.get_offset_text().set_position((1.15, 1.0))
    ax4.tick_params(axis="y", labelcolor=theta_dot_color)
    (line4,) = ax4.plot(
        timestamps,
        delta_theta_dot,
        label=r"$\Delta \dot{\Theta}$",
        color=theta_dot_color,
        linewidth="0.5",
    )

    fig.legend(handles=[line1, line2, line3, line4])
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_total_system_state(
    title: str,
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
    plt.title(title)
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

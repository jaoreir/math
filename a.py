import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp


# System parameters
xg = -1  # ground level
t_max = 2.0
g = 9.81

t_t = 1.5
x_t = 0.8

# Generate multiple trajectories
initial_conditions = [
    (0, 0.0),  # Start at rest
    (0.5, 3.0),  # Medium initial velocity
    (1, 5.0),  # Higher initial velocity
    (1.5, -2.0),  # Initial downward velocity
]

# Position [left, bottom, width, height]
ax_t_t = plt.axes([0.25, 0.1, 0.65, 0.03])
t_t_slider = Slider(ax_t_t, "Target Time", 0.0, 2.0, valinit=t_t, valstep=0.01)
ax_x_t = plt.axes([0.25, 0.5, 0.65, 0.03])
x_t_slider = Slider(ax_x_t, "Target Pos", xg, 2.0, valinit=x_t, valstep=0.01)


def get_bounce_time(x0, v0, xg, g=9.81):
    # xg = x + vt - 1/2*g*t^2
    # (-1/2*g)t^2 + v*t + (x - xg) = 0
    a = -0.5 * g
    b = v0
    c = x0 - xg

    determinate = b * b - 4 * a * c
    if determinate < 0:
        # No real solution
        return None

    # Assuming g>0, always take negative root
    t_b = (-b - math.sqrt(determinate)) / (2 * a)
    return t_b if t_b > 0 else None


def get_bounce_pos(x0, v0, t_t, xg, g=9.81):
    t_curr = 0
    pos = x0
    vel = v0

    while t_curr < t_t:
        dt = get_bounce_time(pos, vel, xg, g)
        if dt is None or t_curr + dt > t_t:
            # No more bounce possible, evaluate final position
            dt = t_t - t_curr
            return pos + vel * dt - 0.5 * g * dt * dt
        else:
            # Update to bounce event
            pos = xg
            vel = vel - g * dt
            vel = -vel  # Bounce
            t_curr += dt


def calc_init_velocity(x0, x_t, t_t, xg, g=9.81, eps=1e-6, max_iter=500):
    # Initial guess: 0 velo to minimize kinetic energy
    v0 = 0
    delta = 1e-5

    # Newton's method
    for _ in range(max_iter):
        x = get_bounce_pos(x0, v0, t_t, xg, g)
        error = x - x_t

        # Found solution
        if abs(error) < 1e-3:
            print(f"Found at {v0}, error={error}")
            return v0

        # Calculate derivative around x
        x_plus = get_bounce_pos(x0, v0 + delta, t_t, xg, g)
        derivative = (x_plus - x) / delta

        if abs(derivative) < eps:
            print("Derivative too small to continue!")
            return 0

        v0 -= error / derivative

    print("max iterations reached without convergence")
    return 0


def phase_space_trajectory(x0, v0, t_max, xg, g=9.81, dt=0.01):
    """Compute phase space trajectory with bounces"""
    times = [0]
    positions = [x0]
    velocities = [v0]

    t = 0
    x = x0
    v = v0

    while t < t_max:
        # Time to next bounce or final time
        # Solve quadratic: x + vt - 0.5gt² = xg
        a = -0.5 * g
        b = v
        c = x - xg

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            # No more bounces, just evolve to final time
            ts = np.arange(t, t_max, dt)
            xs = x + v * (ts - t) - 0.5 * g * (ts - t) ** 2
            vs = v - g * (ts - t)

            times.extend(ts[1:])
            positions.extend(xs[1:])
            velocities.extend(vs[1:])
            break

        dt_bounce = (-b - np.sqrt(discriminant)) / (2 * a)
        if t + dt_bounce > t_max:
            # Evolve to final time without bounce
            ts = np.arange(t, t_max, dt)
            xs = x + v * (ts - t) - 0.5 * g * (ts - t) ** 2
            vs = v - g * (ts - t)

            times.extend(ts[1:])
            positions.extend(xs[1:])
            velocities.extend(vs[1:])
            break

        # Evolve to bounce
        ts = np.arange(t, t + dt_bounce, dt)
        xs = x + v * (ts - t) - 0.5 * g * (ts - t) ** 2
        vs = v - g * (ts - t)

        times.extend(ts[1:])
        positions.extend(xs[1:])
        velocities.extend(vs[1:])

        # Update for bounce
        t = t + dt_bounce
        x = xg
        v = -(v - g * dt_bounce)  # Elastic bounce

        times.append(t)
        positions.append(x)
        velocities.append(v)

    return np.array(times), np.array(positions), np.array(velocities)


# Create visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))


colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions)))

# Energy contours
x_grid = np.linspace(-2, 4, 100)
v_grid = np.linspace(-10, 10, 100)
X, V = np.meshgrid(x_grid, v_grid)
E = 0.5 * V**2 + g * X  # Energy per unit mass

def draw_stuff(x_t, t_t):
    ax1.cla()
    ax2.cla()
    ax3.cla()
    # Plot phase space trajectories
    for (x0, v0), color in zip(initial_conditions, colors):
        v0 = calc_init_velocity(x0, x_t, t_t, xg)

        t, x, v = phase_space_trajectory(x0, v0, t_max, xg)

        # Plot x(t)
        # ax1.plot(t, x, color=color, label=f"v₀={v0}")
        ax1.plot(t, x, color=color)

        # Plot phase space
        ax2.plot(x, v, color=color)
        # Mark initial point
        ax2.scatter([x0], [v0], color=color, s=100, marker="o")
        # Mark bounce points
        bounce_mask = np.abs(x - xg) < 1e-6
        ax2.scatter(x[bounce_mask], v[bounce_mask],
                    color=color, s=50, marker="x")

        # Plot v(t)
        ax3.plot(t, v, color=color)

        # Plot estimated bounce position
        bounce_pos = get_bounce_pos(x0, v0, t_t, xg, g)
        ax1.scatter(t_t, bounce_pos, color=color)

    # Formatting
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Position")
    ax1.grid(True)
    # ax1.legend()
    ax1.axhline(y=xg, color="k", linestyle="--", alpha=0.5)
    ax1.set_title("Position vs Time")

    ax2.set_xlabel("Position")
    ax2.set_ylabel("Velocity")
    ax2.grid(True)
    ax2.axvline(x=xg, color="k", linestyle="--", alpha=0.5)
    ax2.set_title("Phase Space")

    ax2.contour(X, V, E, levels=10, alpha=0.2, colors="gray")

    ax3.set_xlabel("Time")
    ax3.set_ylabel("Velocity")
    ax3.grid(True)
    ax3.set_title("Velocity vs Time")


def update(val):
    t_t = t_t_slider.val
    x_t = x_t_slider.val
    draw_stuff(x_t, t_t)
    fig.canvas.draw_idle()


t_t_slider.on_changed(update)
x_t_slider.on_changed(update)

update(None)
plt.tight_layout()
plt.show()

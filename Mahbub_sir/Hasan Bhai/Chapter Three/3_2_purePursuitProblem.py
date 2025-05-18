import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
fighter_speed = 20
bomber_speed = 15
time_step = 1
end_time = 30
shoot_range = 5

fighter_start = (0, 50)
fighter_x, fighter_y = fighter_start
bomber_start = (100, 0)
bomber_end = (200, 100)
bomber_x, bomber_y = bomber_start

fighter_x_positions = [fighter_x]
fighter_y_positions = [fighter_y]
bomber_x_positions = [bomber_x]
bomber_y_positions = [bomber_y]

hit = False
hit_position = (0, 0)
def update_positions():
    global fighter_x, fighter_y, bomber_x, bomber_y, hit, hit_position

    distance = np.hypot(bomber_x - fighter_x, bomber_y - fighter_y)

    if distance < shoot_range:
        hit = True
        hit_position = (fighter_x, fighter_y)
        return
    cosine = (bomber_x - fighter_x) / distance
    sine = (bomber_y - fighter_y) / distance

    fighter_x += fighter_speed * cosine * time_step
    fighter_y += fighter_speed * sine * time_step

    if bomber_x < bomber_end[0]:
        bomber_x += bomber_speed * time_step
    if bomber_y < bomber_end[1]:
        bomber_y += bomber_speed * 0.5 * time_step

    # Append new positions to lists
    fighter_x_positions.append(fighter_x)
    fighter_y_positions.append(fighter_y)
    bomber_x_positions.append(bomber_x)
    bomber_y_positions.append(bomber_y)

fig, ax = plt.subplots()
ax.set_xlim(-50, 300)
ax.set_ylim(-50, 200)

# Fighter and bomber paths
fighter_path, = ax.plot([], [], 'b-', label="Fighter Path")
bomber_path, = ax.plot([], [], 'r-', label="Bomber Path")
fighter_dot, = ax.plot([], [], 'bo')
bomber_dot, = ax.plot([], [], 'ro')
hit_marker, = ax.plot([], [], 'go', markersize=10, label="Hit Position")  # Mark hit position

def init():
    fighter_path.set_data([], [])
    bomber_path.set_data([], [])
    fighter_dot.set_data([], [])
    bomber_dot.set_data([], [])
    hit_marker.set_data([], [])
    return fighter_path, bomber_path, fighter_dot, bomber_dot, hit_marker
def animate(frame):
    global hit
    if hit:
        print("Hit! The fighter has hit the bomber.")
        fighter_dot.set_color('green')
        hit_marker.set_data([hit_position[0]], [hit_position[1]])
        ani.event_source.stop()
        return fighter_path, bomber_path, fighter_dot, bomber_dot, hit_marker

    update_positions()
    fighter_path.set_data(fighter_x_positions, fighter_y_positions)
    bomber_path.set_data(bomber_x_positions, bomber_y_positions)
    fighter_dot.set_data([fighter_x_positions[-1]], [fighter_y_positions[-1]])
    bomber_dot.set_data([bomber_x_positions[-1]], [bomber_y_positions[-1]])

    return fighter_path, bomber_path, fighter_dot, bomber_dot, hit_marker
ani = FuncAnimation(fig, animate, frames=np.arange(0, end_time, time_step), init_func=init, blit=True)

ax.legend()
plt.title("Pure Pursuit Problem: Fighter Chasing Bomber with Fixed Start and End Positions")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid()
plt.show()

import math
def simulate_projectile(v0, el, k, m, n):

    theta = math.radians(el)
    x = 0
    y = 0
    g = 9.8
    dt = 0.01

    trajectory_x = [x]
    trajectory_y = [y]

    while y >= 0:

        dv = -g * math.sin(theta) - (k * math.pow(v0, n) / m)
        dtheta = -g * math.cos(theta) / v0

        dx = v0 * math.cos(theta) * dt
        dy = v0 * math.sin(theta) * dt

        v0 += dv * dt
        theta += dtheta * dt
        x += dx
        y += dy

        trajectory_x.append(x)
        trajectory_y.append(y)

    return trajectory_x, trajectory_y

v0 = 100
el = 45
k = 0.1
m = 30
n = 0.001

trajectory_x, trajectory_y = simulate_projectile(v0, el, k, m, n)

for i in range(len(trajectory_x)):
    print(f"Point {i + 1}: x = {trajectory_x[i]}, y = {trajectory_y[i]}")
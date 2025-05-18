import random
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return x**3

n = 0
m = 0

x_min, x_max = 2.0, 5.0
y_min, y_max = 0, 170

for i in range(300):
    rx = random.uniform(x_min, x_max)
    ry = random.uniform(y_min, y_max)
    n += 1
    if ry <= func(rx):
        m += 1
        plt.scatter(rx, ry, color="green", s=20, label="Under the curve" if m == 1 else "")
    else:
        plt.scatter(rx, ry, color="blue", s=20, label="Above the curve" if n - m == 1 else "")

x_vals = np.linspace(x_min, x_max, 500)
y_vals = func(x_vals)

plt.plot(x_vals, y_vals, color='red', label='y = x^3', linewidth=2)

plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color='black', linestyle='--', label="Bounding Box", linewidth=1.5)

area_of_rectangle = (x_max - x_min) * (y_max - y_min)
estimated_area = (m / n) * area_of_rectangle  # Estimate of area under the curve

plt.title("Monte Carlo Simulation with Random Points Under y = x³", fontsize=14)
plt.xlabel("X-axis (range: 2.0 to 5.0)", fontsize=12)
plt.ylabel("Y-axis (range: 0 to 180)", fontsize=12)

plt.legend()

plt.grid(True)

plt.annotate(f'Total Points: {n}', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=12, ha='center')
plt.annotate(f'Points Under Curve: {m}', xy=(0.5, 0.85), xycoords='axes fraction', fontsize=12, ha='center')
plt.annotate(f'Estimated Area Under Curve: {estimated_area:.2f}', xy=(0.5, 0.8), xycoords='axes fraction', fontsize=12, ha='center')

plt.show()

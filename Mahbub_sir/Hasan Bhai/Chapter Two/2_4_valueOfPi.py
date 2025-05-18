import math
import random
import matplotlib.pyplot as plt

def func(x):
    val = 1 - (x ** 2)
    val = math.sqrt(val) if val >= 0 else 0
    return val

m = 0
n = 0

for i in range(280):
    rx = random.randint(0, 1000) / 1000
    ry = random.randint(0, 1000) / 1000
    n += 1

    if ry <= func(rx):
        m += 1
        plt.scatter(rx, ry, color="green")
    else:
        plt.scatter(rx, ry, color="blue")

x = []
y = []
i = 0
h = 0.01
while i <= 1:
    x.append(i)
    y.append(round(func(i), 5))
    i += h
    i = round(i, 2)

plt.plot(x, y, color="red", linewidth=2, label='y = sqrt(1 - x^2)')

area = (m / n) * 4
actual_area = math.pi
error = abs(actual_area - area)
error_percentage = (error / actual_area) * 100

plt.title("Monte Carlo Simulation to Estimate Area of a Quarter Circle", fontsize=14)
plt.xlabel("X-axis (0 to 1)", fontsize=12)
plt.ylabel("Y-axis (0 to 1)", fontsize=12)

plt.annotate(f'Total Points: {n}', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=12, ha='center')
plt.annotate(f'Points Under Curve: {m}', xy=(0.5, 0.85), xycoords='axes fraction', fontsize=12, ha='center')
plt.annotate(f'Estimated Area: {area:.4f}', xy=(0.5, 0.8), xycoords='axes fraction', fontsize=12, ha='center')
plt.annotate(f'Error: {error_percentage:.2f}%', xy=(0.5, 0.75), xycoords='axes fraction', fontsize=12, ha='center')

plt.legend()

plt.grid(True)

plt.show()

print("Estimated Area:", area)
print("Error:", error_percentage, "%")
print("Points Under Curve:", m)
print("Total Points:", n)

import random
import numpy as np
import matplotlib.pyplot as plt

length = 1000
height = 600

deviation_x = length / 2
deviation_y = height / 2
def getRNN():
    rnn = np.random.randn()
    return rnn

HIT = 0
N = 0

plt.figure(figsize=(8, 6))

for i in range(100):
    x = getRNN() * deviation_x
    y = getRNN() * deviation_y
    N += 1

    if (-length/2 <= x <= length/2) and (-height/2 <= y <= height/2):
        HIT += 1
        plt.scatter(x, y, color="green", label="Hit" if i == 0 else "")
    else:
        plt.scatter(x, y, color="blue", label="Miss" if i == 0 else "")
area_x = [-500, 500, 500, -500, -500]
area_y = [-300, -300, 300, 300, -300]
plt.plot(area_x, area_y, color="red", label="Boundary")

plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Random Points and Hits within a Rectangular Area")
plt.legend()

accuracy = (HIT / N) * 100
plt.text(-450, 320, f"Accuracy: {accuracy:.2f}%", fontsize=12, color="black")

plt.show()

print(f"Hits: {HIT}, Total: {N}")
print(f"Accuracy: {accuracy:.2f}%")

import numpy as np
import matplotlib.pyplot as plt

# Get user input for number of points
n = int(input("Enter the number of random points: "))

# Generate random (x, y) points in the range [-1, 1] for a unit circle
x_rand = np.random.uniform(-1, 1, n)
y_rand = np.random.uniform(-1, 1, n)

# Check if points are inside the unit circle
inside_circle = x_rand**2 + y_rand**2 <= 1

# Estimate the value of Pi
pi_estimate = (np.sum(inside_circle) / n) * 4

# Output result
print(f"Estimated value of Pi: {pi_estimate:.5f}")

# Plot the simulation
plt.figure(figsize=(6, 6))
plt.scatter(x_rand[inside_circle], y_rand[inside_circle], color='blue', s=1, label='Inside Circle')
plt.scatter(x_rand[~inside_circle], y_rand[~inside_circle], color='red', s=1, label='Outside Circle')

# Draw the unit circle
circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
plt.gca().add_patch(circle)

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.title(f'Monte Carlo Estimation of Pi\nEstimated Pi = {pi_estimate:.5f}')
plt.legend()
plt.show()

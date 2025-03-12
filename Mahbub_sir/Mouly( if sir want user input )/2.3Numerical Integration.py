import numpy as np
import matplotlib.pyplot as plt

# Get user input for function
expr = input("Enter the function in terms of x (e.g., x**2 + sin(x)): ")

# Define the function
f = lambda x: eval(expr, {"x": x, "np": np})

# Get integration limits
a = float(input("Enter lower limit of integration: "))
b = float(input("Enter upper limit of integration: "))
n = int(input("Enter number of random points: "))

# Generate random x values
gx = np.linspace(a, b, 1000)
y_vals = f(gx)

# Define y limits for random sampling
y_min, y_max = min(0, min(y_vals)), max(y_vals)
rand_x = np.random.uniform(a, b, n)
rand_y = np.random.uniform(y_min, y_max, n)

# Count points below the function curve
below_curve = rand_y < f(rand_x)
integral_estimate = (b - a) * (y_max - y_min) * (np.sum(below_curve) / n)

# Output results
print(f"Estimated integral: {integral_estimate:.5f}")

# Plot the function and points
plt.figure(figsize=(8, 6))
plt.plot(gx, y_vals, 'r', label=f'f(x) = {expr}')
plt.scatter(rand_x[below_curve], rand_y[below_curve], color='blue', s=5, label='Points below')
plt.scatter(rand_x[~below_curve.astype(bool)], rand_y[~below_curve.astype(bool)], color='green', s=5, label='Points above')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'Monte Carlo Integration\nEstimated Integral = {integral_estimate:.5f}')
plt.legend()
plt.show()

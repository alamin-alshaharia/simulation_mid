import random
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# User input
num_bombs = int(input("Enter the number of bombs: "))
sigma_x = float(input("Enter standard deviation for x (e.g., 500): "))
sigma_y = float(input("Enter standard deviation for y (e.g., 300): "))

# Constants
mu_x, mu_y = 0, 0  # Mean (center of depot)

# Store results
results = []
coordinates = []

for i in range(1, num_bombs + 1):
    # Generate normal random numbers
    z_x = np.random.normal(0, 1)
    z_y = np.random.normal(0, 1)

    # Compute strike coordinates
    x = mu_x + sigma_x * z_x
    y = mu_y + sigma_y * z_y

    # Determine hit or miss
    hit = (-sigma_x <= x <= sigma_x) and (-sigma_y <= y <= sigma_y)
    result = "Hit" if hit else "Miss"

    # Store results
    results.append([i, round(z_x, 2), round(x, 2), round(z_y, 2), round(y, 2), result])
    coordinates.append((x, y, hit))

# Print table
headers = ["Bomb Strike", "RNN (z_x)", "x (m)", "RNN (z_y)", "y (m)", "Result"]
print("\nSimulation of Bombing Operation:\n")
print(tabulate(results, headers=headers, tablefmt="grid"))

# Compute and print hit percentage
hits = sum(1 for r in results if r[-1] == "Hit")
hit_percentage = (hits / num_bombs) * 100
print(f"\nTotal Hits: {hits} out of {num_bombs}")
print(f"Hit Percentage: {hit_percentage:.2f}%")


# Plot results
def plot_bombing(coordinates):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw depot area (hit zone)
    depot = plt.Rectangle((-sigma_x, -sigma_y), sigma_x*2, sigma_y*2, color='gray', alpha=0.3, label='Depot Area')
    ax.add_patch(depot)

    # Plot bombs
    for x, y, hit in coordinates:
        color = 'green' if hit else 'red'
        ax.scatter(x, y, c=color, label='Hit' if hit else 'Miss', s=50)

    ax.set_xlim(-3 * sigma_x, 3 * sigma_x)
    ax.set_ylim(-3 * sigma_y, 3 * sigma_y)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.title("Bomb Strike Simulation")
    plt.legend(['Depot Area', 'Hit', 'Miss'])
    plt.grid(True)
    plt.show()


plot_bombing(coordinates)

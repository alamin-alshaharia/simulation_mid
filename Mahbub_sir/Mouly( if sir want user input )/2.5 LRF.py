import numpy as np
import matplotlib.pyplot as plt
import time


def get_probabilities():
    while True:
        try:
            forward_prob = float(input("Enter probability for moving Forward: "))
            left_prob = float(input("Enter probability for moving Left: "))
            right_prob = float(input("Enter probability for moving Right: "))

            total = forward_prob + left_prob + right_prob

            if abs(total - 1.0) < 0.01:
                return [forward_prob, left_prob, right_prob]
            else:
                print("Probabilities must sum to 1. Please try again.")
        except ValueError:
            print("Invalid input. Please enter numeric values.")


# Take user input for the number of steps
steps = int(input("Enter the number of steps for the random walk: "))

# Get movement probabilities
probabilities = get_probabilities()

# Initialize starting position
x, y = 0, 0

# Store the path for visualization
x_path, y_path = [x], [y]

# Print table header
print(f"\n{'Step':<6} {'Random Number':<15} {'Direction':<10} {'Position (x,y)':<15}")
print("=" * 50)

# Set up the figure for live updating
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(6, 6))

# Perform the random walk
for step in range(1, steps + 1):
    rand_num = np.random.rand()  # Generate a random number between 0 and 1

    # Determine movement based on probabilities
    if rand_num < probabilities[0]:  # Move Forward
        direction = "Forward"
        y += 1
    elif rand_num < probabilities[0] + probabilities[1]:  # Move Left
        direction = "Left"
        x -= 1
    else:  # Move Right
        direction = "Right"
        x += 1

    # Store path
    x_path.append(x)
    y_path.append(y)

    # Print step details in table format
    print(f"{step:<6} {rand_num:<15.8f} {direction:<10} ({x}, {y})")

    # Plot the updated graph
    ax.clear()
    ax.plot(x_path, y_path, marker='o', linestyle='-', markersize=4, label="Random Walk Path")
    ax.scatter([0], [0], color='green', s=100, label="Start (0,0)")  # Start position
    ax.scatter([x], [y], color='red', s=100, label=f"Step {step} ({x},{y})")  # Current position
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Random Walk - Step {step}/{steps}")
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.legend()
    ax.grid()

    plt.draw()
    plt.pause(0.5)  # Pause for animation effect

# Keep the final plot open
plt.ioff()
plt.show()
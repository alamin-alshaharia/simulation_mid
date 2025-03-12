import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Random walk - Probability forward , backward , left , right = 40% 10% 25% 25%
random_walk_probabilities = [0.4, 0.1, 0.25, 0.25]

# Generate random digit assignments
random_walk_random_digits = []
count = 0
for i in range(len(random_walk_probabilities)):
    random_walk_random_digits.append([])
    for j in range(int(random_walk_probabilities[i] * 100)):
        random_walk_random_digits[i].append(count)
        count += 1

# Set up the distances in the x and y axes
up, down, left, right = 75, 45, 60, 60

# Initialize the starting point as (0,0)
x, y = 0, 0

# Create empty lists to store the x and y coordinates of the movement
x_list = [x]
y_list = [y]

# Take user input for the number of steps
num_steps = int(input("Enter the number of steps: "))

# Generate a random direction and move accordingly
for i in range(num_steps):  # move user-defined times
    step = random.randint(0, 99)
    direction = None

    # Find direction from generated random num
    for j in range(len(random_walk_random_digits)):
        if step in random_walk_random_digits[j]:
            if j == 0:
                direction = "forward"
            elif j == 1:
                direction = "backward"
            elif j == 2:
                direction = "left"
            elif j == 3:
                direction = "right"
            break

    if direction == 'forward':
        y += up
    elif direction == 'backward':
        y -= down
    elif direction == 'left':
        x -= left
    else:
        x += right
    x_list.append(x)
    y_list.append(y)

# Calculate the total distance covered
distance = math.sqrt(x ** 2 + y ** 2)

# Create a scatter plot with green points
plt.scatter(x_list, y_list, color='green')
# Create a line plot with blue lines connecting the points
plt.plot(x_list, y_list, color='blue')
# Add the first point as a cyan point
plt.scatter(x_list[0], y_list[0], color='cyan', label='Start')
# Add the last point as a pink point
plt.scatter(x_list[-1], y_list[-1], color='red', label='End')
plt.title(f"Total Distance Covered: {distance:.2f} cm")
plt.xlabel("X-axis (cm)")
plt.ylabel("Y-axis (cm)")
plt.legend()
plt.show()

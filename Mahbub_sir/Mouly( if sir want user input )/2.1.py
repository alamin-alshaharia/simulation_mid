import random
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

# Define an irregular polygon shape
irregular_shape = Polygon([(1, 1), (4, 1), (5, 3), (3, 5), (2, 4)])

# Get number of random samples from the user
num_samples = int(input("Enter the number of random points to sample: "))

# Bounding box for the polygon (min_x, min_y, max_x, max_y)
min_x, min_y, max_x, max_y = irregular_shape.bounds

# Generate random points and check if they are inside the shape
inside_points = []
outside_points = []

for _ in range(num_samples):
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    point = Point(x, y)

    if irregular_shape.contains(point):
        inside_points.append((x, y))
    else:
        outside_points.append((x, y))

# Estimate area
bounding_box_area = (max_x - min_x) * (max_y - min_y)
estimated_area = (len(inside_points) / num_samples) * bounding_box_area

# Print result
print(f"Estimated Area of the Irregular Shape: {estimated_area:.2f}")

# Plot the shape and points
x_poly, y_poly = irregular_shape.exterior.xy
plt.plot(x_poly, y_poly, 'b-', label="Irregular Shape")

# Scatter points
plt.scatter(*zip(*inside_points), color='green', s=2, label="Inside Points")
plt.scatter(*zip(*outside_points), color='red', s=2, label="Outside Points")

plt.legend()
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Monte Carlo Simulation for Area Estimation")
plt.show()

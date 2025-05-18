import numpy as np
import matplotlib.pyplot as plt
def random_walk_single_digit(steps, num_persons):
    positions = [[[0, 0]] for _ in range(num_persons)]

    directions = {'F': [0, 1], 'L': [-1, 0], 'R': [1, 0]}

    for step in range(1, steps + 1):
        for person in range(num_persons):
            rand_num = np.random.randint(10)

            if rand_num < 5:
                direction = 'F'
            elif rand_num < 8:
                direction = 'L'
            else:
                direction = 'R'
            new_position = [positions[person][-1][0] + directions[direction][0],
                            positions[person][-1][1] + directions[direction][1]]
            positions[person].append(new_position)

    return positions

num_steps = 10
num_persons = 5

trajectories = random_walk_single_digit(num_steps, num_persons)

final_distances = [np.linalg.norm(trajectory[-1]) for trajectory in trajectories]

winner_index = np.argmax(final_distances)

plt.figure(figsize=(8, 6))

for person in range(num_persons):
    x_coords = [pos[0] for pos in trajectories[person]]
    y_coords = [pos[1] for pos in trajectories[person]]

    if person == winner_index:
        plt.plot(x_coords, y_coords, marker='o', linestyle='-', label=f'Person {person + 1} (Winner)', color='gold',
                 linewidth=2, markersize=8)
    else:
        plt.plot(x_coords, y_coords, marker='o', linestyle='-', label=f'Person {person + 1}')

plt.title('Random Walk Trajectories')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.grid(True)
plt.legend()
plt.annotate(f'Winner: Person {winner_index + 1}', xy=(0.5, 0.9), xycoords='axes fraction', fontsize=12, ha='center',
             color='gold')
plt.show()
print("Final Distances from Start:", final_distances)
print(f"The winner is Person {winner_index + 1} with a distance of {final_distances[winner_index]:.2f}.")

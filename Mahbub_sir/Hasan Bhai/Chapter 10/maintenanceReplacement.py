import numpy as np
import matplotlib.pyplot as plt

SEED = 12345
np.random.seed(SEED)

# Parameters
run = 2000
cost1 = cost2 = 0
count = kont = 0

# Initialize times for the present policy
t_present = [0] * 5

# Present policy simulation
for i in range(1, 5):
    x = np.random.rand()
    t_present[i] = int(1000 + x * 1000)

clock = 0
present_counts = []

while clock <= run:
    present_counts.append((clock, t_present[1], t_present[2], t_present[3], t_present[4], count))
    small = min(t_present[1:5])
    for i in range(1, 5):
        t_present[i] -= small
    jj = np.argmin(t_present[1:5]) + 1  # Get index of the smallest time
    t_present[jj] = int(1000 + np.random.rand() * 1000)  # Update time for the machine
    clock += small
    count += 1

# Proposed policy simulation
clock = 0
kont = 0
t_proposed = [0] * 5
proposed_counts = []

while clock <= run:
    for i in range(1, 5):
        x = np.random.rand()
        t_proposed[i] = int(1000 + x * 1000)
    proposed_counts.append((clock, t_proposed[1], t_proposed[2], t_proposed[3], t_proposed[4], kont))
    small = min(t_proposed[1:5])
    clock += small
    kont += 1

# Cost calculations
cost1 = count * (200 + 100)
cost2 = kont * 2 * 200 + kont * 4 * 100

# Display costs
print(f"Cost Present policy = {cost1} Cost Proposed policy = {cost2}")

# Prepare data for plotting
present_counts = np.array(present_counts)
proposed_counts = np.array(proposed_counts)

# Plotting the results
plt.figure(figsize=(12, 6))

# Present Policy
plt.subplot(1, 2, 1)
plt.plot(present_counts[:, 0], present_counts[:, 1], label='T1', color='blue')
plt.plot(present_counts[:, 0], present_counts[:, 2], label='T2', color='orange')
plt.plot(present_counts[:, 0], present_counts[:, 3], label='T3', color='green')
plt.plot(present_counts[:, 0], present_counts[:, 4], label='T4', color='red')
plt.title('Present Policy Maintenance Times')
plt.xlabel('Clock Time')
plt.ylabel('Remaining Time')
plt.legend()
plt.grid()

# Proposed Policy
plt.subplot(1, 2, 2)
plt.plot(proposed_counts[:, 0], proposed_counts[:, 1], label='T1', color='blue')
plt.plot(proposed_counts[:, 0], proposed_counts[:, 2], label='T2', color='orange')
plt.plot(proposed_counts[:, 0], proposed_counts[:, 3], label='T3', color='green')
plt.plot(proposed_counts[:, 0], proposed_counts[:, 4], label='T4', color='red')
plt.title('Proposed Policy Maintenance Times')
plt.xlabel('Clock Time')
plt.ylabel('Remaining Time')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

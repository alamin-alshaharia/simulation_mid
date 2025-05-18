import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
run_time = 1000  # Total simulation time
delta_t = 1.0    # Time increment
np.random.seed(0)  # For reproducibility

# Initial values for buffers and sheet inventory
buffers = [0, 0, 0]  # Buffers 1, 2, and 3
sheets = [500, 400, 750, 600]  # Sheet inventories for 4 types
machine_status = [True, True, True, True]  # Machine uptime status
downtimes = [3, 5, 4, 15]  # Initial downtimes for machines
uptimes = [100, 90, 180, 240]  # Uptime durations for machines
buffer_max = [1, 1, 1]  # Maximum buffer capacities
counter = 0  # Production counter
clock = 0.0  # Simulation clock

# Lists to track data for plotting
time_points = []
buffer_status = [[] for _ in range(3)]
sheet_status = [[] for _ in range(4)]
throughput = []

# Simulation loop
while clock < run_time:
    # Update machines based on their uptime and downtime
    for i in range(4):
        if machine_status[i]:  # Machine is up
            machine_status[i] = False  # Set machine down
            downtimes[i] += np.random.uniform(3, 10)  # Random downtime increment
            sheets[i] -= 1  # Reduce sheet inventory
        elif clock >= downtimes[i]:  # Machine is down and downtime has passed
            machine_status[i] = True  # Set machine back up
            downtimes[i] = clock + np.random.exponential(uptimes[i])  # Schedule next downtime

    # Update buffers based on machine operations
    if machine_status[0] and buffers[0] < buffer_max[0]:  # If Shear machine is up
        buffers[0] += 1  # Add to buffer 1
    if machine_status[1] and buffers[1] < buffer_max[1] and buffers[0] > 0:  # If Punch machine is up
        buffers[1] += 1  # Add to buffer 2
        buffers[0] -= 1  # Consume from buffer 1
    if machine_status[2] and buffers[2] < buffer_max[2] and buffers[1] > 0:  # If Form machine is up
        buffers[2] += 1  # Add to buffer 3
        buffers[1] -= 1  # Consume from buffer 2

    # Increment production counter if Bend machine is active
    if machine_status[3]:
        counter += 1

    # Store data for plotting
    time_points.append(clock)
    for j in range(3):
        buffer_status[j].append(buffers[j])
    for j in range(4):
        sheet_status[j].append(sheets[j])
    throughput.append(counter)

    # Increment simulation clock
    clock += delta_t

# Plotting the results
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot buffer levels
axs[0].plot(time_points, buffer_status[0], label="Buffer 1")
axs[0].plot(time_points, buffer_status[1], label="Buffer 2")
axs[0].plot(time_points, buffer_status[2], label="Buffer 3")
axs[0].set_title("Buffer Levels Over Time")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Buffer Level")
axs[0].legend()

# Plot sheet inventory levels
for i in range(4):
    axs[1].plot(time_points, sheet_status[i], label=f"Sheet {i+1} Inventory")
axs[1].set_title("Sheet Inventory Levels Over Time")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Sheet Level")
axs[1].legend()

# Plot throughput
axs[2].plot(time_points, throughput, label="Throughput")
axs[2].set_title("Throughput Over Time")
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Number of Products Produced")
axs[2].legend()

plt.tight_layout()
plt.show()

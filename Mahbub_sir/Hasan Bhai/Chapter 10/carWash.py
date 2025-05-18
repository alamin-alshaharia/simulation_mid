import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
np.random.seed(0)
num_stalls = 3  # Number of washing stalls
queue_capacity = 4  # Max cars in queue
arrival_rate = 1 / 0.3  # Average inter-arrival time rate
service_rate = 1 / 0.5  # Average service time rate
total_time = 5000  # Total simulation time
time_increment = 0.3  # Time increment for each simulation step

# State variables
queue = 0  # Current queue length
arrivals = 0  # Total arrivals
served = 0  # Total served cars
lost = 0  # Total lost cars
clock = 0  # Simulation clock
next_arrival = 0  # Time for next car arrival
stall_end_times = np.zeros(num_stalls)  # Each stall's next available time

# Lists to store data for animation
time_data = []
queue_data = []
lost_data = []

# Main simulation loop
while clock < total_time:
    # Determine if a new car arrives
    if clock >= next_arrival:
        arrivals += 1
        if queue < queue_capacity:
            queue += 1
            # Schedule the next arrival
            inter_arrival_time = np.random.exponential(1 / arrival_rate)
            next_arrival = clock + inter_arrival_time
        else:
            # Lost car if queue is full
            lost += 1
            # Schedule the next arrival
            inter_arrival_time = np.random.exponential(1 / arrival_rate)
            next_arrival = clock + inter_arrival_time

    # Check each stall for availability and serve cars from the queue
    for i in range(num_stalls):
        if clock >= stall_end_times[i] and queue > 0:
            # A car leaves the queue and enters the stall
            queue -= 1
            served += 1

            # Determine the service time based on probability
            r = np.random.rand()
            if r < 0.2:
                service_time = 5.0
            elif r < 0.85:
                service_time = 10.0
            else:
                service_time = 15.0
            stall_end_times[i] = clock + service_time  # Update stall availability

    # Record data for plotting
    time_data.append(clock)
    queue_data.append(queue)
    lost_data.append(lost)

    # Increment simulation clock
    clock += time_increment

# Calculate performance statistics
service_ratio = (served / arrivals) * 100 if arrivals > 0 else 0
lost_ratio = (lost / arrivals) * 100 if arrivals > 0 else 0

# Plotting the queue length and lost cars over time
plt.figure(figsize=(12, 6))

# Queue length plot
plt.subplot(2, 1, 1)
plt.plot(time_data, queue_data, label="Queue Length", color="blue")
plt.xlabel("Time")
plt.ylabel("Queue Length")
plt.title("Queue Length Over Time")
plt.legend()

# Lost cars plot
plt.subplot(2, 1, 2)
plt.plot(time_data, lost_data, label="Cars Lost", color="red")
plt.xlabel("Time")
plt.ylabel("Cars Lost")
plt.title("Lost Cars Over Time")
plt.legend()

plt.tight_layout()
plt.show()

# Display final statistics
print("\nFinal Simulation Statistics:")
print(f"Total Clock Time: {clock:.2f}")
print(f"Total Arrivals: {arrivals}")
print(f"Total Served: {served}")
print(f"Total Lost: {lost}")
print(f"Service Ratio: {service_ratio:.2f}%")
print(f"Lost Ratio: {lost_ratio:.2f}%")

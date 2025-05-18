import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

cumulative_freq = [0.0, 0.05, 0.11, 0.21, 0.35, 0.51, 0.67, 0.80, 0.90, 0.97, 1.00]
capacity = 190
run = 500
total_hours = 0
total_idle = 0
total_waiting = 0
cost_idle = 50
cost_waiting = 100
days = list(range(run))
idle_times = []
waiting_times = []
order_hours = []

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
ax1.set_title("Idle Capacity Over Days")
ax2.set_title("Waiting Time Over Days")
ax3.set_title("Order Hours Over Days")
ax1.set_xlabel("Days")
ax1.set_ylabel("Idle Hours")
ax2.set_xlabel("Days")
ax2.set_ylabel("Waiting Hours")
ax3.set_xlabel("Days")
ax3.set_ylabel("Order Hours")


def simulate(day):
    global total_hours, total_idle, total_waiting, capacity

    r = np.random.rand()
    orders = next(j for j in range(1, 11) if cumulative_freq[j - 1] < r <= cumulative_freq[j])

    hours = sum([(np.random.rand() * 5 + i * 5) for i in np.random.randint(0, 11, orders)])

    idle = max(0, capacity - hours)
    waiting = max(0, hours - capacity)
    total_idle += idle
    total_waiting += waiting
    total_hours += hours

    idle_times.append(idle)
    waiting_times.append(waiting)
    order_hours.append(hours)

    if day % (run // 10) == 0:
        capacity += 5

    ax1.plot(days[:len(idle_times)], idle_times, color='blue')
    ax2.plot(days[:len(waiting_times)], waiting_times, color='orange')
    ax3.plot(days[:len(order_hours)], order_hours, color='green')


ani = animation.FuncAnimation(fig, simulate, frames=run, repeat=False)
plt.tight_layout()
plt.show()

total_cost = total_idle * cost_idle + total_waiting * cost_waiting
print(f"Final Total Cost: {total_cost:.2f}")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

fc = 60000
sp = 10
csr = 0.6
vcl, vc2 = 4.5, 5.5
salel, sale2 = 10000, 12000
sale3, sale4 = 13000, 15000
cost_threshold = 5000

n = int(input("Enter number of trials: "))

profits = []
loss_count = 0
profit_count = 0

fig, ax = plt.subplots()
ax.set_title("Profit Analysis Simulation")
ax.set_xlim(0, n)
ax.set_ylim(-10000, 20000)
line, = ax.plot([], [], lw=2, color="blue", label="Profit")
loss_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
profit_text = ax.text(0.02, 0.8, '', transform=ax.transAxes)
ax.legend()

def simulate(i):
    global loss_count, profit_count

    r1 = np.random.rand()
    vc = vcl + r1 * (vc2 - vcl)

    r2 = np.random.rand()
    if r2 < csr:
        demand = salel + r2 * (sale2 - salel)
    else:
        demand = sale3 + r2 * (sale4 - sale3)

    profit = demand * (sp - vc) - fc
    profits.append(profit)

    if profit < -cost_threshold:
        loss_count += 1
    elif profit > cost_threshold:
        profit_count += 1

    loss_text.set_text(f"Loss > 5000: {loss_count} times")
    profit_text.set_text(f"Profit > 5000: {profit_count} times")

    line.set_data(range(len(profits)), profits)
    return line, loss_text, profit_text 

ani = FuncAnimation(fig, simulate, frames=n, repeat=False, interval=50)

plt.show()

total_profit = sum(profits)
print(f"Total Profit/Loss after {n} trials: {total_profit:.2f}")

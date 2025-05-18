import numpy as np
import matplotlib.pyplot as plt

order_cost = 100
holding_cost = 5
initial_inventory = 10

weeks_to_simulate = 52
order_quantity = 6
reorder_point = 4

demand_levels = [0, 1, 2, 3, 4]
demand_probs = [0.2, 0.5, 0.15, 0.1, 0.05]

lead_time_levels = [1, 2, 3, 4, 5]
lead_time_probs = [0.1, 0.25, 0.5, 0.1, 0.05]


def generate_demand():
    return np.random.choice(demand_levels, p=demand_probs)


def generate_lead_time():
    return np.random.choice(lead_time_levels, p=lead_time_probs)


inventory = initial_inventory
total_holding_cost = 0
total_order_cost = 0
orders = []

inventory_levels = []
holding_costs = []
order_costs = []
total_costs = []

for week in range(weeks_to_simulate):
    if orders:
        orders = [(qty, lead_time - 1) for qty, lead_time in orders if lead_time > 1]
        if any(lead_time == 1 for qty, lead_time in orders):
            inventory += sum(qty for qty, lead_time in orders if lead_time == 1)
            orders = [(qty, lead_time) for qty, lead_time in orders if lead_time != 1]

    demand = generate_demand()
    inventory -= demand
    if inventory < 0:
        inventory = 0

    weekly_holding_cost = holding_cost * inventory
    total_holding_cost += weekly_holding_cost

    if inventory <= reorder_point:
        total_order_cost += order_cost
        lead_time = generate_lead_time()
        orders.append((order_quantity, lead_time))

    total_cost = total_holding_cost + total_order_cost
    inventory_levels.append(inventory)
    holding_costs.append(total_holding_cost)
    order_costs.append(total_order_cost)
    total_costs.append(total_cost)

plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(range(weeks_to_simulate), inventory_levels, marker='o', color='b', label="Inventory Level")
plt.axhline(reorder_point, color='r', linestyle="--", label="Reorder Point")
plt.title("Weekly Inventory Level Over Time")
plt.xlabel("Week")
plt.ylabel("Inventory Level")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(weeks_to_simulate), total_costs, color='g', label="Total Cost")
plt.plot(range(weeks_to_simulate), holding_costs, color='c', linestyle="--", label="Holding Cost Accumulation")
plt.plot(range(weeks_to_simulate), order_costs, color='m', linestyle="--", label="Order Cost Accumulation")
plt.title("Cost Accumulation Over Time")
plt.xlabel("Week")
plt.ylabel("Cost (Rs)")
plt.legend()

plt.tight_layout()
plt.show()

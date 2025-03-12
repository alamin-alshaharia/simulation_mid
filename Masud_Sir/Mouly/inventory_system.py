import simpy
import random

# Simulation Parameters
INITIAL_INVENTORY = 60
DEMAND_SIZES = [1, 2, 3, 4]
DEMAND_PROBABILITIES = [0.167, 0.500, 0.833, 1.000]
MEAN_INTERDEMAND_TIME = 0.10
DELIVERY_LAG_RANGE = (0.50, 1.00)
SIMULATION_LENGTH = 120
ORDER_POLICIES = [(20, 40), (20, 60), (20, 80), (20, 100), (40, 60), (40, 80), (40, 100), (60, 80), (60, 100)]
K = 32.0
i = 3.0
h = 1.0
pi = 5.0


def demand_generator():
    """Generates demand based on probability distribution"""
    r = random.random()
    for idx, prob in enumerate(DEMAND_PROBABILITIES):
        if r < prob:
            return DEMAND_SIZES[idx]
    return DEMAND_SIZES[-1]  # Default to max demand if random value is large


def inventory_system(env, policy, results):
    """Simulates inventory system for given (s, S) policy"""
    s, S = policy
    inventory = INITIAL_INVENTORY
    total_cost, ordering_cost, holding_cost, shortage_cost = 0, 0, 0, 0
    order_arrival_time = None  # Track order arrival

    while env.now < SIMULATION_LENGTH:
        # Process pending order arrival
        if order_arrival_time and env.now >= order_arrival_time:
            inventory += order_quantity
            order_arrival_time = None  # Reset order tracking

        # Generate demand
        demand = demand_generator()
        if inventory >= demand:
            inventory -= demand
        else:
            shortage_cost += (demand - inventory) * pi
            inventory = 0

        # Calculate holding cost
        holding_cost += inventory * h

        # Place order if inventory ≤ reorder point (s)
        if inventory <= s and not order_arrival_time:
            order_quantity = S - inventory
            ordering_cost += K + i * order_quantity
            order_arrival_time = env.now + random.uniform(*DELIVERY_LAG_RANGE)  # Schedule order arrival

        # Wait until the next demand (Exponential time gap)
        yield env.timeout(random.expovariate(1.0 / MEAN_INTERDEMAND_TIME))

    # Store results after simulation completes
    total_cost = ordering_cost + holding_cost + shortage_cost
    results.append((policy, total_cost, ordering_cost, holding_cost, shortage_cost))


def run_simulation():
    """Runs the simulation for all policies"""
    random.seed(42)
    results = []

    for policy in ORDER_POLICIES:
        env = simpy.Environment()  # Create a new environment for each policy
        env.process(inventory_system(env, policy, results))  # Run inventory system
        env.run()  # Execute simulation

    return results


def find_most_efficient_policy(results):
    """Finds the policy with the lowest total cost"""
    most_efficient_policy = min(results, key=lambda x: x[1])  # Get the policy with the lowest total cost
    return most_efficient_policy


def display_results(results):
    """Displays the results in a formatted output"""
    print("Single-product inventory system")
    print(f"Initial inventory level                      {INITIAL_INVENTORY} items")
    print(f"Number of demand sizes                        {len(DEMAND_SIZES)}")
    print("Distribution function of demand sizes    ", "   ".join(f"{p:.3f}" for p in DEMAND_PROBABILITIES))
    print(f"Mean interdemand time                      {MEAN_INTERDEMAND_TIME:.2f} months")
    print(
        f"Delivery lag range                         {DELIVERY_LAG_RANGE[0]:.2f} to {DELIVERY_LAG_RANGE[1]:.2f} months")
    print(f"Length of the simulation                    {SIMULATION_LENGTH} months")
    print(f"K = {K}   i = {i}   h = {h}   pi = {pi}")
    print(f"Number of policies                            {len(ORDER_POLICIES)}")
    print("                  Average        Average        Average        Average")
    print("   Policy       total cost    ordering cost  holding cost   shortage cost")

    # Display all results and highlight the most efficient policy
    most_efficient_policy = find_most_efficient_policy(results)
    for policy, total, order, hold, short in results:
        if policy == most_efficient_policy[0]:
            print(
                f" {policy}         {total:.2f}          {order:.2f}          {hold:.2f}          {short:.2f}  <--- Most Efficient")
        else:
            print(f" {policy}         {total:.2f}          {order:.2f}          {hold:.2f}          {short:.2f}")

    print("\nMost Efficient Policy:")
    print(f"Policy: {most_efficient_policy[0]}")
    print(f"Total Cost: {most_efficient_policy[1]:.2f}")
    print(f"Ordering Cost: {most_efficient_policy[2]:.2f}")
    print(f"Holding Cost: {most_efficient_policy[3]:.2f}")
    print(f"Shortage Cost: {most_efficient_policy[4]:.2f}")


if __name__ == "__main__":
    results = run_simulation()
    display_results(results)

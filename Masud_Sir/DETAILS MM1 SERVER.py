import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq


class MM1QueueSimulation:
    def __init__(self, mean_interarrival_time, mean_service_time, max_customers):
        """
        Initialize the M/M/1 queue simulation with parameters.

        Args:
            mean_interarrival_time: Mean time between customer arrivals (lambda^-1)
            mean_service_time: Mean service time (mu^-1)
            max_customers: Maximum number of customers to simulate
        """
        self.mean_interarrival_time = mean_interarrival_time
        self.mean_service_time = mean_service_time
        self.max_customers = max_customers

        # Performance metrics
        self.total_delay = 0
        self.total_customers_served = 0
        self.total_queue_length = 0
        self.num_queue_samples = 0
        self.server_busy_time = 0

        # Simulation state
        self.current_time = 0
        self.queue = deque()
        self.server_busy = False
        self.server_finish_time = float('inf')

        # Event queue (priority queue based on event time)
        self.event_queue = []

        # Statistics tracking
        self.queue_length_history = []
        self.waiting_times = []
        self.time_history = []

    def generate_interarrival_time(self):
        """Generate a random interarrival time using exponential distribution."""
        return np.random.exponential(self.mean_interarrival_time)

    def generate_service_time(self):
        """Generate a random service time using exponential distribution."""
        return np.random.exponential(self.mean_service_time)

    def schedule_arrival(self):
        """Schedule the next customer arrival."""
        interarrival_time = self.generate_interarrival_time()
        arrival_time = self.current_time + interarrival_time
        heapq.heappush(self.event_queue, (arrival_time, "arrival"))

    def process_arrival(self):
        """Process a customer arrival event."""
        # Schedule the next arrival if we haven't reached max customers
        if self.total_customers_served < self.max_customers:
            self.schedule_arrival()

        # If server is idle, start service immediately
        if not self.server_busy:
            self.server_busy = True
            service_time = self.generate_service_time()
            self.server_finish_time = self.current_time + service_time
            heapq.heappush(self.event_queue, (self.server_finish_time, "departure"))
            self.total_customers_served += 1
        else:
            # Server is busy, add customer to queue
            self.queue.append(self.current_time)  # Store arrival time for delay calculation

        # Record queue length statistics
        self.queue_length_history.append((self.current_time, len(self.queue)))

    def process_departure(self):
        """Process a customer departure event."""
        if self.queue:
            # Get next customer from queue
            arrival_time = self.queue.popleft()
            waiting_time = self.current_time - arrival_time
            self.total_delay += waiting_time
            self.waiting_times.append(waiting_time)

            # Serve the next customer
            service_time = self.generate_service_time()
            self.server_finish_time = self.current_time + service_time
            heapq.heappush(self.event_queue, (self.server_finish_time, "departure"))
            self.total_customers_served += 1
        else:
            # No customers in queue, server becomes idle
            self.server_busy = False
            self.server_finish_time = float('inf')

        # Record queue length statistics
        self.queue_length_history.append((self.current_time, len(self.queue)))

    def run_simulation(self):
        """Run the complete simulation."""
        # Schedule the first arrival
        self.schedule_arrival()

        # Previous time for tracking server utilization
        prev_time = 0

        # Run until we've served the maximum number of customers
        while self.total_customers_served < self.max_customers:
            # Get next event
            event_time, event_type = heapq.heappop(self.event_queue)

            # Update server busy time
            if self.server_busy:
                self.server_busy_time += event_time - prev_time

            # Update current time
            prev_time = self.current_time
            self.current_time = event_time

            # Sample queue length for average calculation
            time_diff = self.current_time - prev_time
            self.total_queue_length += len(self.queue) * time_diff
            self.num_queue_samples += time_diff

            # Process the event
            if event_type == "arrival":
                self.process_arrival()
            elif event_type == "departure":
                self.process_departure()

            # Record time for plotting
            self.time_history.append(self.current_time)

    def get_results(self):
        """Calculate and return simulation results."""
        avg_delay = self.total_delay / self.max_customers if self.max_customers > 0 else 0
        avg_queue_length = self.total_queue_length / self.num_queue_samples if self.num_queue_samples > 0 else 0
        server_utilization = self.server_busy_time / self.current_time if self.current_time > 0 else 0

        return {
            "Average Delay in Queue": avg_delay,
            "Average Number in Queue": avg_queue_length,
            "Server Utilization": server_utilization,
            "Time Simulation Ended": self.current_time,
            "Total Customers Served": self.total_customers_served,
            "Queue Length History": self.queue_length_history,
            "Waiting Times": self.waiting_times,
            "Time History": self.time_history
        }

    def plot_queue_length(self):
        """Plot the queue length over time."""
        times, lengths = zip(*self.queue_length_history)

        plt.figure(figsize=(10, 6))
        plt.step(times, lengths, where='post')
        plt.title('Queue Length over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Customers in Queue')
        plt.grid(True)
        plt.show()

    def plot_waiting_time_histogram(self):
        """Plot histogram of customer waiting times."""
        plt.figure(figsize=(10, 6))
        plt.hist(self.waiting_times, bins=30, alpha=0.7, color='blue')
        plt.title('Distribution of Customer Waiting Times')
        plt.xlabel('Waiting Time')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()


# Example usage
def run_mm1_simulation(mean_interarrival, mean_service, max_customers):
    """Run an M/M/1 queue simulation with specified parameters."""
    print(f"Running M/M/1 Queue Simulation:")
    print(f"Mean interarrival time: {mean_interarrival}")
    print(f"Mean service time: {mean_service}")
    print(f"Maximum customers: {max_customers}")
    print("-" * 50)

    # Calculate theoretical results
    rho = mean_service / mean_interarrival
    if rho < 1:
        theoretical_avg_queue = (rho ** 2) / (1 - rho)
        theoretical_avg_delay = mean_service * rho / (1 - rho)
        print(f"Theoretical Results (if ρ < 1):")
        print(f"Traffic intensity (ρ): {rho:.4f}")
        print(f"Theoretical average queue length: {theoretical_avg_queue:.4f}")
        print(f"Theoretical average delay: {theoretical_avg_delay:.4f}")
        print("-" * 50)
    else:
        print("Warning: Traffic intensity ρ ≥ 1. Queue theoretically unstable.")
        print("-" * 50)

    # Run simulation
    sim = MM1QueueSimulation(mean_interarrival, mean_service, max_customers)
    sim.run_simulation()
    results = sim.get_results()

    # Print results
    print("Simulation Results:")
    print(f"Average Delay in Queue: {results['Average Delay in Queue']:.4f}")
    print(f"Average Number in Queue: {results['Average Number in Queue']:.4f}")
    print(f"Server Utilization: {results['Server Utilization']:.4f}")
    print(f"Time Simulation Ended: {results['Time Simulation Ended']:.4f}")
    print(f"Total Customers Served: {results['Total Customers Served']}")

    # Plot results
    sim.plot_queue_length()
    sim.plot_waiting_time_histogram()

    return results


# Example parameters
mean_interarrival_time = 1.0  # Average time between arrivals
mean_service_time = 0.8  # Average service time
max_customers = 1000  # Maximum number of customers to simulate

# Run the simulation
results = run_mm1_simulation(mean_interarrival_time, mean_service_time, max_customers)
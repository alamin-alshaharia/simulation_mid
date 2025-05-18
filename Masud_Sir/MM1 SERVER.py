import heapq
import numpy as np

class MM1QueueSimulation:
    def __init__(self, mean_interarrival_time, mean_service_time, max_customers):
        # Initialize simulation parameters
        self.mean_interarrival_time = mean_interarrival_time  # Average time between customer arrivals
        self.mean_service_time = mean_service_time  # Average time it takes to serve a customer
        self.max_customers = max_customers  # Total number of customers to simulate
        self.total_delay = 0  # Accumulated waiting time of customers in the queue
        self.total_customers_served = 0  # Count of customers who have been served
        self.total_queue_length = 0  # Accumulated queue length over time
        self.num_queue_samples = 0  # Total time duration for which queue length is sampled
        self.server_busy_time = 0  # Total time the server is busy
        self.current_time = 0  # Current simulation time
        self.queue = []  # List representing the queue (FIFO)
        self.server_busy = False  # Flag indicating if the server is currently busy
        self.server_finish_time = float('inf')  # Time when the server will finish serving the current customer
        self.event_queue = []  # Priority queue (heap) for event scheduling

    def generate_interarrival_time(self):
        # Generate interarrival time using exponential distribution
        return np.random.exponential(self.mean_interarrival_time)

    def generate_service_time(self):
        # Generate service time using exponential distribution
        return np.random.exponential(self.mean_service_time)

    def schedule_arrival(self):
        # Schedule the next customer arrival event
        interarrival_time = self.generate_interarrival_time()
        heapq.heappush(self.event_queue, (self.current_time + interarrival_time, "arrival"))

    def process_arrival(self):
        # Handle customer arrival event
        if self.total_customers_served < self.max_customers:
            self.schedule_arrival() #schedule the next arrival if we are still under the max customer limit.
        if not self.server_busy:
            # If the server is idle, start serving the customer
            self.server_busy = True
            service_time = self.generate_service_time()
            self.server_finish_time = self.current_time + service_time
            heapq.heappush(self.event_queue, (self.server_finish_time, "departure"))
            self.total_customers_served += 1
        else:
            # If the server is busy, add the customer to the queue
            self.queue.append(self.current_time)

    def process_departure(self):
        # Handle customer departure event
        if self.queue:
            # If there are customers in the queue, serve the next one
            arrival_time = self.queue.pop(0)
            waiting_time = self.current_time - arrival_time
            self.total_delay += waiting_time
            service_time = self.generate_service_time()
            self.server_finish_time = self.current_time + service_time
            heapq.heappush(self.event_queue, (self.server_finish_time, "departure"))
            self.total_customers_served += 1
        else:
            # If the queue is empty, the server becomes idle
            self.server_busy = False
            self.server_finish_time = float('inf')

    def run_simulation(self):
        # Run the simulation
        self.schedule_arrival()  # Schedule the first arrival
        prev_time = 0
        while self.total_customers_served < self.max_customers:
            # Process events until the maximum number of customers is served
            event_time, event_type = heapq.heappop(self.event_queue)
            if self.server_busy:
                # Accumulate the server busy time
                self.server_busy_time += event_time - prev_time
            prev_time = self.current_time
            self.current_time = event_time
            # Accumulate the queue length over time
            self.total_queue_length += len(self.queue) * (event_time - prev_time)
            self.num_queue_samples += event_time - prev_time
            if event_type == "arrival":
                self.process_arrival()
            elif event_type == "departure":
                self.process_departure()

    def get_results(self):
        # Calculate and return simulation results
        avg_delay = self.total_delay / self.max_customers if self.max_customers > 0 else 0
        avg_queue_length = self.total_queue_length / self.num_queue_samples if self.num_queue_samples > 0 else 0
        server_utilization = self.server_busy_time / self.current_time if self.current_time > 0 else 0
        return {
            "Average Delay in Queue": avg_delay,
            "Average Number in Queue": avg_queue_length,
            "Server Utilization": server_utilization,
            "Time Simulation Ended": self.current_time,
            "Total Customers Served": self.total_customers_served
        }

# Example usage
mean_interarrival_time = 1.0
mean_service_time = 0.8
max_customers = 1000

sim = MM1QueueSimulation(mean_interarrival_time, mean_service_time, max_customers)
sim.run_simulation()
results = sim.get_results()

print("Simulation Results:")
print(f"Average Delay in Queue: {results['Average Delay in Queue']:.4f}")
print(f"Average Number in Queue: {results['Average Number in Queue']:.4f}")
print(f"Server Utilization: {results['Server Utilization']:.4f}")
print(f"Time Simulation Ended: {results['Time Simulation Ended']:.4f}")
print(f"Total Customers Served: {results['Total Customers Served']}")
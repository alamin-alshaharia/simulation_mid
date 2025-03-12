import math
import random

# Constants and variables initialization
Q_LIMIT = 100
BUSY = 1
IDLE = 0
next_event_type, num_custs_delayed, num_delays_required, num_events, num_in_q, server_status = 0, 0, 0, 2, 0, IDLE
area_num_in_q, area_server_status, mean_interarrival, mean_service, sim_time, total_of_delays = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
time_arrival = [0.0] * (Q_LIMIT + 1)
time_last_event, time_next_event = 0.0, [0.0] * 3

# Function to generate exponential random variates (using Python's random module)
def expon(mean):
    return -mean * math.log(random.random())

# Function to initialize simulation parameters
def initialize():
    global sim_time, server_status, num_in_q, time_last_event, num_custs_delayed, total_of_delays, area_num_in_q, area_server_status, time_next_event
    global time_last_event  # Add global declaration for time_last_event

    # Initialize simulation parameters and schedule the first arrival event
    sim_time = 0.0
    server_status = IDLE
    num_in_q = 0
    time_last_event = 0.0
    num_custs_delayed = 0
    total_of_delays = 0.0
    area_num_in_q = 0.0
    area_server_status = 0.0
    time_next_event[1] = sim_time + expon(mean_interarrival)
    time_next_event[2] = 1.0e+30

# Function to determine the next event type and update the simulation time
def timing():
    global next_event_type, sim_time
    min_time_next_event = 1.0e+29
    next_event_type = 0

    # Find the next event type with the smallest time in time_next_event
    for i in range(1, num_events + 1):
        if time_next_event[i] < min_time_next_event:
            min_time_next_event = time_next_event[i]
            next_event_type = i

    # If no event is scheduled, terminate the simulation
    if next_event_type == 0:
        print("\nEvent list empty at time", sim_time)
        exit(1)

    # Update the simulation time to the time of the next event
    sim_time = min_time_next_event

# Function to handle the arrival of a customer
def arrive():
    global num_in_q, server_status, num_custs_delayed, total_of_delays
    delay = 0.0

    # Schedule the next arrival event
    time_next_event[1] = sim_time + expon(mean_interarrival)

    # If the server is busy, the arriving customer joins the queue
    if server_status == BUSY:
        num_in_q += 1
        # Check if the queue length exceeds the limit
        if num_in_q > Q_LIMIT:
            print("\nOverflow of the array time_arrival at time", sim_time)
            exit(2)
        # Store the arrival time of the customer in the queue
        time_arrival[num_in_q] = sim_time
    else:
        # If the server is idle, the arriving customer starts service immediately
        delay = 0.0
        total_of_delays += delay
        num_custs_delayed += 1
        server_status = BUSY
        # Schedule the departure event for the current customer
        time_next_event[2] = sim_time + expon(mean_service)

# Function to handle the departure of a customer
def depart():
    global num_in_q, server_status, num_custs_delayed, total_of_delays
    delay = 0.0

    # If there are no customers in the queue, the server becomes idle and no departure event is scheduled
    if num_in_q == 0:
        server_status = IDLE
        time_next_event[2] = 1.0e+30
    else:
        # If there are customers in the queue, the first customer departs
        num_in_q -= 1
        # Calculate the delay experienced by the departing customer and update the total delays
        delay = sim_time - time_arrival[1]
        total_of_delays += delay
        num_custs_delayed += 1
        # Schedule the departure event for the next customer in the queue
        time_next_event[2] = sim_time + expon(mean_service)
        # Move the remaining customers' arrival times in the queue one position forward
        for i in range(1, num_in_q + 1):
            time_arrival[i] = time_arrival[i + 1]

# Function to generate the simulation report
def report():
    global area_num_in_q, area_server_status, num_custs_delayed, sim_time

    print("\n\nSingle-server queueing system\n")
    print("Mean interarrival time{:11.3f} ".format(mean_interarrival))
    print("Mean service time{:16.3f} ".format(mean_service))
    print("Number of customers{:14d}".format(num_delays_required))

    # Calculate and print the average delay in the queue if there are delayed customers
    if num_custs_delayed > 0:
        print("\n\nAverage delay in queue{:11.3f} ".format(total_of_delays / num_custs_delayed))
    else:
        print("\n\nAverage delay in queue: N/A (No customers have been delayed)")

    # Calculate and print the average number of customers in the queue and server utilization
    if sim_time > 0:
        print("Average number in queue{:10.3f}".format(area_num_in_q / sim_time))
        print("Server utilization{:15.3f}".format(area_server_status / sim_time))
        print("Time simulation ended{:12.3f} ".format(sim_time))
    else:
        # If simulation has not started, print N/A for average number in queue, server utilization, and simulation end time
        print("Average number in queue: N/A (Simulation has not started)")
        print("Server utilization: N/A (Simulation has not started)")
        print("Time simulation ended: N/A (Simulation has not started)")

# Function to update time-average statistics during the simulation
def update_time_avg_stats():
    global area_num_in_q, area_server_status, time_last_event  # Add global declaration for time_last_event
    time_since_last_event = sim_time - time_last_event
    time_last_event = sim_time
    # Update the area for the number of customers in the queue and the server status based on the time since the last event
    area_num_in_q += num_in_q * time_since_last_event
    area_server_status += server_status * time_since_last_event

# Main function to run the simulation
def main():
    global num_custs_delayed, outfile

    # Input parameters (You can change these values as desired)
    global mean_interarrival, mean_service, num_delays_required
    mean_interarrival = 5.0
    mean_service = 2.0
    num_delays_required = 1000

    # Initialize the simulation
    initialize()

    # Run the simulation while more delays are still needed
    while num_custs_delayed < num_delays_required:
        timing()
        update_time_avg_stats()
        if next_event_type == 1:
            arrive()
        elif next_event_type == 2:
            depart()

    # Generate the report and close the output file
    report()

if __name__ == "__main__":
    main()
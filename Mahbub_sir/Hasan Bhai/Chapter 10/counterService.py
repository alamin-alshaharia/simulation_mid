import numpy as np
import matplotlib.pyplot as plt

SEED = 12345
np.random.seed(SEED)


def get_interarrival_time(p, at):
    r = np.random.rand()
    for i in range(len(p) - 1):
        if p[i] < r <= p[i + 1]:
            return at[i]
    return at[-1]


def get_service_time(f, t):
    r = np.random.rand()
    for i in range(len(f) - 1):
        if f[i] < r <= f[i + 1]:
            return t[i]
    return t[-1]


def coffee_house_simulation(run, num_servers):
    n = 6
    m = 5
    p = [0.05, 0.40, 0.65, 0.80, 0.90, 0.97, 1.0]
    at = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    f = [0.05, 0.30, 0.65, 0.85, 1.0]
    t = [1, 2, 3, 4, 5]

    avg_waiting_time = 0

    for _ in range(run):
        se = [0] * num_servers
        nat = 0.0
        cwt = 0.0
        counter = 0

        for _ in range(run):
            iat = get_interarrival_time(p, at)
            st = get_service_time(f, t)
            nat += iat

            min_time = min(se)
            k = se.index(min_time)

            if nat < min_time:
                wt = min_time - nat
                se[k] = min_time + st
            else:
                wt = 0
                se[k] = nat + st

            cwt += wt
            counter += 1

        avg_waiting_time += cwt / counter

    avg_waiting_time /= run
    return avg_waiting_time


def main():
    run = int(input("Enter the number of runs: "))
    server_counts = range(2, 6)
    avg_waiting_times = []

    for j in server_counts:
        avg_waiting_time = coffee_house_simulation(run, j)
        avg_waiting_times.append(avg_waiting_time)
        print(f"Servers: {j}, Total arrivals: {run}, Average Waiting Time: {avg_waiting_time:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(server_counts, avg_waiting_times, marker='o')
    plt.title("Average Waiting Time vs Number of Servers")
    plt.xlabel("Number of Servers")
    plt.ylabel("Average Waiting Time")
    plt.xticks(server_counts)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

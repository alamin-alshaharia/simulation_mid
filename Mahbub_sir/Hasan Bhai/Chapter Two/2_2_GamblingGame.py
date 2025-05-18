import random
import matplotlib.pyplot as plt
def play_game():
    lose_money = 0
    head_count = 0
    tail_count = 0
    while abs(head_count - tail_count) < 3:
        lose_money += 1
        if random.randint(0, 9) < 5:
            head_count += 1
        else:
            tail_count += 1

    return lose_money

n_simulations = 100

flips_per_game = [play_game() for _ in range(n_simulations)]

running_avg = [sum(flips_per_game[:i+1]) / (i+1) for i in range(n_simulations)]
overall_avg = sum(flips_per_game) / n_simulations

plt.figure(figsize=(12, 6))
plt.plot(flips_per_game, label="Flips per Game", color='blue', marker='o', markersize=4)
plt.plot(running_avg, label="Running Average of Flips", color='red', linestyle='dashed', linewidth=2)

plt.axhline(overall_avg, color='purple', linestyle='dashed', linewidth=2, label=f"Overall Average Flips = {overall_avg:.2f}")

plt.axhline(8, color='green', linestyle='dashed', linewidth=2, label="Gain per Game = 8 Flips")

plt.title("Step-by-Step Simulation: Number of Flips per Game", fontsize=16)
plt.xlabel("Game Number", fontsize=12)
plt.ylabel("Number of Flips", fontsize=12)
plt.legend()
plt.grid(True)

plt.show()

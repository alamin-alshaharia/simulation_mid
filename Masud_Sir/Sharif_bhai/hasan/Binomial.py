import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

n = int(input("Enter the number of trials (n): "))
p = float(input("Enter the probability of success in each trial (p): "))

x = np.arange(0, n + 1)

binom_pmf = binom.pmf(x, n, p)

print("\nBinomial Distribution Probabilities:")
for k, prob in enumerate(binom_pmf):
    print(f"P(X = {k}) = {prob:.4f}")

plt.bar(x, binom_pmf, color='skyblue', edgecolor='black')
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.show()

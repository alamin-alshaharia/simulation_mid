import numpy as np
from scipy.stats import chi2

seed = 1
a = 1664525
m = 2**32
N = 1000

random_numbers = []
X = seed
for _ in range(N):
    X = (a * X) % m
    random_numbers.append(X / m)

k = 10
observed_counts, _ = np.histogram(random_numbers, bins=k, range=(0, 1))
expected_count = N / k

chi_square_stat = np.sum((observed_counts - expected_count) ** 2 / expected_count)

alpha = 0.05
critical_value = chi2.ppf(1 - alpha, k - 1)

print("Chi-square Statistic:", chi_square_stat)
print("Critical Value:", critical_value)
print("Passes Chi-square Test:", chi_square_stat < critical_value)

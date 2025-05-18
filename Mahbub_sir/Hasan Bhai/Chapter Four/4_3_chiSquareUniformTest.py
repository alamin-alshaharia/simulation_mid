from scipy.stats import chi2

ri = [12]
a = 13
b = 1
m = 97
n = 100

i = 1
while i <= n:
    rn = ((a * ri[i - 1]) + b) % m
    ri.append(rn)
    i += 1

ri = ri[1:]

print("Generated random numbers (first 10):", ri[:10])
print("Total generated numbers:", len(ri))

cls = list(range(0, 101, 10))
print("Class intervals:", cls)

frequency = [0] * (len(cls) - 1)

for i in ri:
    for j in range(len(cls) - 1):
        if cls[j] < i <= cls[j + 1]:
            frequency[j] += 1

print("Frequencies for each class interval:", frequency)

expected_frequency = n / 10
print("Expected frequency per class:", expected_frequency)

dif_sq = 0
for f in frequency:
    x = (f - expected_frequency) ** 2
    dif_sq += x

chi_sqr = dif_sq / expected_frequency
print("Chi-square statistic:", chi_sqr)

df = len(frequency) - 1
alpha = 0.05
critical_value = chi2.ppf(1 - alpha, df)

print("Degrees of freedom:", df)
print("Critical value at alpha = 0.05:", critical_value)

if chi_sqr < critical_value:
    print("Result: Accept the null hypothesis. The random numbers are uniformly distributed.")
else:
    print("Result: Reject the null hypothesis. The random numbers are not uniformly distributed.")

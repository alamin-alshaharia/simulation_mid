import numpy as np
import matplotlib.pyplot as plt

p = 0.4
n = 1000

data = np.random.binomial(n=1, p=p, size=n)

unique, counts = np.unique(data, return_counts=True)
frequencies = dict(zip(unique, counts))

plt.figure(figsize=(8, 5))
plt.bar(frequencies.keys(), frequencies.values(), color=['blue', 'orange'], width=0.4)
plt.xticks([0, 1], labels=['Failure (0)', 'Success (1)'])
plt.xlabel("Outcome")
plt.ylabel("Frequency")
plt.title(f"Bernoulli Distribution (p={p}) for {n} Trials")
plt.show()

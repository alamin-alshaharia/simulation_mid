
r = [1]
a = 13
b = 1
m = 97
n = 10


for i in range(1, n + 1):
    rn = (a * r[i - 1] + b) % m
    r.append(rn)

print("Generated random numbers:", r[1:])

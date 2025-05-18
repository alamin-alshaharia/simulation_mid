
n = 20
seed = 5673

var = int(len(str(seed)) / 2)

rn = []
for i in range(n):
    x = seed ** 2
    st = str(x).zfill(2 * len(str(seed)))
    dig = int(len(st) / 2)
    r = st[dig - var: dig + var]
    seed = int(r)
    rn.append(seed)

print("Generated random numbers:", rn)

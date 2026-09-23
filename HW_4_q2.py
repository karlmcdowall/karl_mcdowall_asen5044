import numpy as np

r1_values = [1, 2, 3, 4, 5, 6]
r2_values = [1, 2, 3, 4, 5, 6]

jpdt = np.zeros((6, 6))
total_states = 0

for r1 in r1_values:
    for r2 in r2_values:
        x = abs(r1 - r2)
        y = max(r1, r2)
        jpdt[x][y - 1] += 1
        total_states += 1

print(
    f"Note that these values still need to be divided by {total_states} to give the probabilities..."
)
print(f"JPDT:\n{jpdt}")

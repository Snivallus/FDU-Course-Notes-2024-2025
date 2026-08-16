import numpy as np

A = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

print(A)
print(np.linalg.inv(A))

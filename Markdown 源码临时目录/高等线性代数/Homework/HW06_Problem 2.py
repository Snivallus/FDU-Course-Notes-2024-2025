import numpy as np

# Define matrices A and B
Y = np.array([
    [1, 0, 0],
    [1, 1, 0],
    [1, 2, 1],
    [1, 3, 3],
    [1, 4, 6]
])

# Perform matrix multiplication
X = np.dot(Y.T,Y)
X = np.linalg.inv(X)
print(X)
X = Y @ X @ X @ Y.T
print(175 * 175 * X)
print(175 * 175)
A = Y @ Y.T
print(A)
print(np.linalg.norm(A @ X @ A - A, ord='fro'))
print(np.linalg.norm(X @ A @ X - X, ord='fro'))
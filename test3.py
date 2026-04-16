import numpy as np

A = np.array([
    [19, -4, -9, -1],
    [-2, 20, -2, -7],
    [6, -5, -25, 9],
    [0, -3, -9, 13]
], dtype=float)

b = np.array([100, -5, 34, 78], dtype=float)

x = np.linalg.solve(A, b)
residual = A @ x - b

print("Точное решение через numpy.linalg.solve:")
print(x)
print("\nНевязка:")
print(residual)
print(np.linalg.norm(residual, np.inf))
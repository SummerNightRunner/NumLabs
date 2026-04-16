import numpy as np

A = np.array([
    [2, 1, 1, 1],
    [4, -6, 0, 1],
    [-2, 7, 2, 2],
    [1, 0, 3, -1]
], dtype=float)

b = np.array([5, -2, 9, 1], dtype=float)

x = np.linalg.solve(A, b)
det = np.linalg.det(A)
inv_A = np.linalg.inv(A)
residual = A @ x - b

print("Решение x:")
print(x)
print("\nОпределитель det(A):")
print(det)
print("\nОбратная матрица A^-1:")
print(inv_A)
print("\nНевязка A*x - b:")
print(residual)
import numpy as np

A = np.array([
    [-11, -9,  0,  0,  0],
    [  5, -15, -2, 0,  0],
    [  0, -8, 11, -3,  0],
    [  0,  0,  6, -15, 4],
    [  0,  0,  0,  3,  6]
], dtype=float)

d = np.array([-122, -48, -14, -50, 42], dtype=float)

x = np.linalg.solve(A, d)
det = np.linalg.det(A)
inv_A = np.linalg.inv(A)
residual = A @ x - d

print("Решение x:")
print(x)
print("\nОпределитель det(A):")
print(det)
print("\nОбратная матрица A^-1:")
print(inv_A)
print("\nНевязка A*x - d:")
print(residual)
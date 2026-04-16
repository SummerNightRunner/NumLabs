import numpy as np

A = np.array([
    [4, 1, 2],
    [1, 3, 1],
    [2, 1, 5]
], dtype=float)

eigenvalues, eigenvectors = np.linalg.eigh(A)

idx = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Собственные значения:")
print(eigenvalues)
print("\nСобственные векторы:")
print(eigenvectors)

print("\nПроверка A*v = lambda*v:")
for i in range(len(eigenvalues)):
    left = A @ eigenvectors[:, i]
    right = eigenvalues[i] * eigenvectors[:, i]
    error = np.linalg.norm(left - right)
    print(f"Вектор {i+1}: ошибка = {error:.2e}")
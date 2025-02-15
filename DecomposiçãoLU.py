import numpy as np

def decomposicao_lu(A):
    """ Realiza a decomposição LU de uma matriz A """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Decomposição em U
        for k in range(i, n):
            soma = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - soma

        # Decomposição em L
        for k in range(i, n):
            if i == k:
                L[i][i] = 1  # Diagonal da L é 1
            else:
                soma = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (A[k][i] - soma) / U[i][i]

    return L, U

def resolucao_lu(L, U, b):
    """ Resolve o sistema Ly = b e depois Ux = y """
    n = L.shape[0]
    y = np.zeros(n)
    x = np.zeros(n)

    # Resolução de Ly = b
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]

    # Resolução de Ux = y
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]

    return x

# Exemplo de uso
A = np.array([[4, 3, 0],
              [3, 4, -1],
              [0, -1, 4]], dtype=float)
b = np.array([24, 30, -24], dtype=float)

# Decomposição LU
L, U = decomposicao_lu(A)
print("Matriz L:\n", L)
print("Matriz U:\n", U)

# Resolução do sistema
solucao = resolucao_lu(L, U, b)
print("Solução do sistema:", solucao)

import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.animation as animation
import time

def matriz(d, alpha, N):
    """
    
    """
    # A U
    offset = [-1, 0, 1]
    up = np.ones(N - 1) * d * alpha
    ppal = - 2 * np.ones(N) * (1 - 4 * d * alpha)
    down = up
    k = np.array([down, ppal, up])
    A = sparse.diags(k, offset)
    A = A.toarray()
    # Condiciones de borde periódicas en ambos extremos.
    A[0][N - 2] = d * alpha
    A[N - 1][1] = d * alpha
    # U B
    upB = np.ones(N - 1) * d * alpha
    ppalB = np.zeros(N)
    downB = upB
    k = np.array([downB, ppalB, upB])
    B = sparse.diags(k, offset)
    B = B.toarray()
    # Condiciones de borde periódicas en ambos extremos.
    A[0][2] = d * alpha
    A[N - 1][N - 2] = d * alpha
    return A, B


def solve(U, V, d, a, b, gamma, Nt):
    """
    Calcula y guarda la solución en tiempo final.
    flag: boolean, true para condiciones de borde periódicas y false para flujo nulo en el borde.
    """
    N = len(U)
    dt = 1 / Nt
    dx = 1 / N
    alpha = dt / (dx ** 2)
    Au, Bu =  matriz(d, alpha, N)
    Av, Bv =  matriz(1, alpha, N)
    # Cantidad máxima de iteraciones
    ti = time.time()
    for i in np.arange(1, Nt + 1):
        U = np.dot(Au, U) + np.dot(U, Bu) + dt * gamma * (a - U + U**2 * V)
        V = np.dot(Av, U) + np.dot(U, Bv) + dt * gamma * (b - U**2 * V)
        if i % 1000 == 0:
            print('avance: ' + str(100 * i / Nt) + ' %')
    tf = time.time()
    print('min: ', (tf - ti) / 60)
    
    return U, V

N = 80
Nt = 10 * 3
u0 = np.random.random((N,N))
v0 = np.random.random((N,N))

solve(U=u0, V=v0, d=10, a=0.1, b=0.9, gamma=2000, Nt=Nt)
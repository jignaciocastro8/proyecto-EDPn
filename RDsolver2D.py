import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from scipy import sparse
from scipy.sparse import linalg
import time


"""
Esta clase se encarga de resolver la edp (4.1) en dos dimensiones.
Se hace uso de un método explícito con actualización matricial. 
"""

class RDsolver2D:
    def __init__(self, Nt, u0, v0, d, a, b, gamma):
        """
        Nt: Tamaño partición temporal. (Cantidad de iteraciones)
        u0: numpy array (N x N), solución u en t = 0.
        v0: numpy array (N x N), solución v en t = 0.
        d: Constante del problema.
        a: Constante del problema.
        gamma: Constante del problema.
        """
        self.u0 = u0
        self.v0 = v0
        self.Nt = Nt
        self.N = len(u0)
        self.d = d
        self.a = a
        self.b = b
        self.gamma = gamma
        # Soluciones.
        self.U = u0
        self.V = v0
        
    
    def matriz(self):
        """
        Crea y retorna la matrices asociadas a la discretización de derivadas.
        """
        offset = [-1, 0, 1]
        # Matriz auxiliar
        up = np.ones(self.N - 1) 
        ppal = - 2 * np.ones(self.N) 
        down = up
        k = np.array([down, ppal, up])
        A = sparse.diags(k, offset)
        A = A.toarray()
        A[0][self.N - 1] = 1
        A[self.N - 1][0] = 1
        return A


    def solve(self):
        """
        Calcula y guarda la solución como matriz de (Nt + 1) x (N + 1).  
        """
        U = self.U
        V = self.V
        alpha = self.N ** 2 / self.Nt
        c = self.d * alpha
        dt = 1 / self.Nt
        gamma = self.gamma
        a = self.a
        b = self.b
        A =  self.matriz()
        T = 50000   
        for _ in np.arange(1, T + 1):
            U = U + alpha * (np.dot(A, U) + np.dot(U, A)) + dt * gamma * (a - U + U**2 * V)
            V = V + self.d * alpha * (np.dot(A, V) + np.dot(V, A)) + dt * gamma * (b - U**2 * V)
        self.U = U
        self.V = V

    def getInit(self):
        return self.u0, self.v0

    
    def plot(self):
        """
        Realiza un plot de la solución.
        """
    
        
        f, axs = plt.subplots(2,2)
        axs[0, 0].matshow(self.u0, cmap='gray')
        axs[0, 1].matshow(self.v0, cmap='gray')
        axs[1, 0].matshow(self.U, cmap='gray')
        axs[1, 1].matshow(self.V, cmap='gray')
        #plt.show()
        """sp =  f.add_subplot(1, 2, 1 )
        plt.title('u')
        for n in np.arange(0, self.Nt, 100):
            plt.plot(u[n])

        sp =  f.add_subplot(1, 2, 2 )
        plt.title('v')
        for n in np.arange(0, self.Nt, 100):
            plt.plot(v[n])
        """



"""TEST"""


# Mesh
dx = 0.1
x = np.arange(dx, 1, dx)
y = np.arange(dx, 1, dx)
xx, yy = np.meshgrid(x, y, sparse=False, indexing='ij')


# Función indicatríz de un sub-cuadradito en el cuadradro [0, 1]x[0, 1]
def f0(x, y):
    return (0.4 <= x <= 0.6) and (0.4 <= y <= 0.6) 

# Gaussiana
g0 = lambda x,y : np.exp(-((x - 0.5)**2 + (y - 0.5)**2))
    



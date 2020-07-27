import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from scipy import sparse
from scipy.sparse import linalg
import time

class RDsolver:
    def __init__(self, L, Nt, N, u0, v0, d, a):
        """
        L: Largo intervalo espacial.
        Nt: Tamaño partición temporal.
        N: Tamaño partición espacial.
        u0: Arreglo, solución u en t = 0.
        v0: Arreglo, solución v en t = 0.
        d: Constante del problema.
        a: Función parámetro del problema.
        """
        self.L = L
        self.Nt = Nt
        self.N = N
        self.u0 = u0
        self.v0 = v0
        self.d = d
        self.a = a
        self.alpha = N**2 / Nt
        # Soluciones.
        self.u = np.zeros((Nt + 1, N + 1))
        self.v = np.zeros((Nt + 1, N + 1))
  
    def matriz(self):
        """
        Crea y retorna la matríz asociada a la discretización de derivadas.
        """
        # Matriz asociada a u.
        up_u = np.ones(self.N - 2) * self.alpha * self.d
        ppal_u = np.ones(self.N - 1) * (1 - 2 * self.alpha * self.d)
        down_u = up_u
        k_u = np.array([down_u, ppal_u, up_u])
        # Matriz asociada a v.
        up_v = np.ones(self.N - 2) * self.alpha 
        ppal_v = np.ones(self.N - 1) * (1 - 2 * self.alpha)
        down_v = up_v
        k_v = np.array([down_v, ppal_v, up_v])
        offset = [-1, 0, 1]
        return sp.sparse.diags(k_u, offset).toarray(), sp.sparse.diags(k_v, offset).toarray()

    def solve(self):
        """
        Calcula y guarda la solución como matriz de (Nt + 1) x (N + 1).  
        """
        a = self.a
        dt = 1 / self.Nt
        u = np.zeros((self.Nt + 1, self.N - 1))
        v = np.zeros((self.Nt + 1, self.N - 1))
        # Tiempo inicial.
        u[0] = self.u0[1:self.N]
        v[0] = self.v0[1:self.N]
        # Matrices.
        U , V = self.matriz()
        for n in np.arange(1, self.Nt + 1):
            u[n] = np.dot(U, u[n - 1]) + dt * u[n - 1] * (v[n - 1] -  self.a(u[n - 1], v[n - 1]))
            v[n] = np.dot(V, v[n - 1]) - dt * u[n - 1] * v[n - 1]
        #arr = np.array([np.zeros(self.Nt + 1)])
        #sol = np.concatenate((arr.T, sol, arr.T), 1)
        self.u = u
        self.v = v 
    
    def plot(self):
        """
        Realiza un plot de la solución.
        """
        
        u = self.u
        v = self.v
        f = plt.figure(facecolor='w', edgecolor='k')
        sp =  f.add_subplot(1, 2, 1 )
        plt.title('u')
        for n in np.arange(0, self.Nt, 100):
            plt.plot(u[n])

        sp =  f.add_subplot(1, 2, 2 )
        plt.title('v')
        for n in np.arange(0, self.Nt, 100):
            plt.plot(v[n])

        plt.show()


# En 1 dimensión.
Nt = 10000
L = 10
N = int(np.sqrt(Nt / 2))
dx = (2*L) / N
f_init = lambda t: t < -L/2
#u0 = 1 / (1 + np.exp(-np.arange(-L, L + dx, dx)**2))
u0 = f_init(np.arange(-L, L + dx, dx))
v0 = np.ones(N + 1)
a_1 = 1 / 2400
a_2 = 1 / 120
d = 1
a = lambda u,v : 1 / ( (1 + u / a_1) * (1 + v / a_2) )
solver = RDsolver(L, Nt, N, u0, v0, d, a)
solver.solve()
solver.plot()


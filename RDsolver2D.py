import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from scipy import sparse
from scipy.sparse import linalg
from matplotlib import animation
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
        # Cond iniciales.
        self.u0 = u0
        self.v0 = v0
        # Parámetros.
        self.Nt = Nt
        self.N = len(u0)
        self.d = d
        self.a = a
        self.b = b
        self.gamma = gamma
        # Soluciones.
        self.U = u0
        self.V = v0
        # Animación
        self.M = []
    
        
    
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
        dt = 1 / self.Nt
        d = self.d
        a = self.a
        b = self.b
        gamma = self.gamma
        A =  self.matriz()
        # Cantidad máxima de iteraciones.
        # Revisar condiciones periódicas.
        T = 100000  
        for i in np.arange(1, T + 1):
            U = U + alpha * (np.dot(A, U) + np.dot(U, A)) + dt * gamma * (a - U + U**2 * V)
            V = V + d * alpha * (np.dot(A, V) + np.dot(V, A)) + dt * gamma * (b - U**2 * V)
            if i % 10 == 0:
                self.M.append(U)
        self.U = U
        self.V = V

    def getAnimation(self):
        return self.M

    def plot(self):
        """
        Realiza un plot
        """
        f, axs = plt.subplots(2,2)
        axs[0, 0].matshow(self.u0)
        axs[0, 1].matshow(self.v0, cmap='rainbow')
        axs[1, 0].matshow(self.U)
        axs[1, 1].matshow(self.V, cmap='rainbow')




"""TEST"""

N = 50

A1 = np.zeros((N, N))
A1[10:20, 10:20] = 1
A1[30:40, 30:40] = 1
B1 = 1 - A1

g = lambda x,y : np.exp(-((x - 3)**2 + (y - 2)**2))
dx = 0.1
x = np.arange(dx, 5, dx)
y = np.arange(dx, 5, dx)
xx, yy = np.meshgrid(x, y, sparse=False, indexing='ij')

u0 = g(xx, yy)
 #np.zeros((N, N))
#u0[20:30, 20:30] = 1
v0 = 1 - u0

solver = RDsolver2D(400000, A1, B1, 10, 0.1, 0.9, 1000)
#solver2 = RDsolver2D(400000, u0, v0, 10, 0.1, 0.9, 1000)

#ti = time.time()
solver.solve()
#solver2.solve()
#tf = time.time()
#print('Tiempo (seg): ', tf - ti)
#solver.plot()
#solver2.plot()
#plt.show() 


"""Aminación"""

M = solver.getAnimation()

def update(i):
    matrice.set_array(M[i])

fig, ax = plt.subplots()
matrice = ax.matshow(np.ones((50, 50)))
plt.colorbar(matrice)

ani = animation.FuncAnimation(fig, update, frames=80000, interval=20)
ani.save('rd1.mp4')
plt.show()





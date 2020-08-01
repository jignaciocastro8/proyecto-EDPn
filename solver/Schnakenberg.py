import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from scipy import sparse
from scipy.sparse import linalg
from matplotlib import animation
import time


"""
Esta clase se encarga de resolver el modelo de Schnakenberg con diferencias finitas.
Se hace uso de un método explícito DF con actualización matricial. 
Asume que la grilla es de tamaño N x N dado por las matrices de condiciones iniciales. 
"""

class Schnakenberg:
    def __init__(self, Nt, u0, v0, d, a, b, gamma, tMax = 100000):
        """
        Nt: Tamaño partición temporal. (Cantidad de iteraciones)
        u0: numpy array (N x N), solución u en t = 0.
        v0: numpy array (N x N), solución v en t = 0.
        d: Constante del problema.
        a: Constante del problema.
        gamma: Constante del problema.
        tMax: cantidad máxima de iteraciones.
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
        self.tMax = tMax
        # Soluciones.
        self.U = u0
        self.V = v0
        self.suma = np.zeros((self.N, self.N))
        # Matriz para animación.
        self.M = []
        # Promedio de la solución sobre la grilla.
        self.mean = 0
    
        
    
    def matriz(self):
        """
        Crea y retorna la matrices asociadas a la discretización de derivadas.
        """
        offset = [-1, 0, 1]
        up = np.ones(self.N - 1) 
        ppal = - 2 * np.ones(self.N) 
        down = up
        k = np.array([down, ppal, up])
        A = sparse.diags(k, offset)
        A = A.toarray()
        # Condiciones de borde periódicas en ambos extremos.
        A[0][self.N - 1] = 1
        A[self.N - 1][0] = 1
        return A


    def solve(self):
        """
        Calcula y guarda la solución en tiempo final.  
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
        T = self.tMax 
        ti = time.time()
        for i in np.arange(1, T + 1):
            U = U + alpha * (np.dot(A, U) + np.dot(U, A)) + dt * gamma * (a - U + U**2 * V)
            V = V + d * alpha * (np.dot(A, V) + np.dot(V, A)) + dt * gamma * (b - U**2 * V)
            #if i % 10 == 0:
            #    self.M.append(U)
        tf = time.time()
        print('min: ', (tf - ti) / 60)
        self.U = U
        self.V = V
        self.mean = [np.mean(U), np.mean(V)]

    def getAnimationMatrix(self):
        """
        Retorna (cuando se calcula) arreglo con matrices con soluciones en varios tiempos.
        """
        return self.M

    def getMean(self):
        """
        Getter del promedio de las soluciones sobre la grilla.
        """
        return self.mean

    def getSol(self):
        """
        Retorna la solución en el último tiempo calculado.
        """
        return self.U, self.V
    
    def setTmax(self, tMax):
        """
        Setter de tMax.
        """
        self.tMax = tMax

    def setNt(self, Nt):
        """
        Setter de Nt.
        """
        self.Nt = Nt
    

    def plot(self):
        """
        Realiza un plot
        """
        f, axs = plt.subplots(1,2)

        axs[0].matshow(self.U, cmap='Blues')
        axs[0].set_title(r'$u$ con $\gamma=$' + str(self.gamma), y = 1, fontname='serif')

        axs[1].matshow(self.V, cmap='Blues')
        axs[1].set_title(r'$v$ con $\gamma=$' + str(self.gamma), y = 1, fontname='serif')







"""TEST"""

# Constructor --> __init__(self, Nt, u0, v0, d, a, b, gamma, tMax = 10000)

"""N = 80

A1 = np.zeros((N, N))
A1[10:20, 10:20] = 1
A1[30:40, 30:40] = 1
B1 = 1 - A1

g = lambda x,y : np.exp(-((x - 3)**2 + (y - 2)**2))
dx = 0.1
x = np.arange(dx, 5, dx)
y = np.arange(dx, 5, dx)
xx, yy = np.meshgrid(x, y, sparse=False, indexing='ij')

#u0 = g(xx, yy)
#np.zeros((N, N))
#u0[20:30, 20:30] = 1
u0 = np.random.random((N, N))
v0 = 1 - u0
# Figure 1 (dos bloques)
#solver = Schnakenberg(300000, A1, B1, 10, 0.1, 0.9, 1000, 50000)
# Figure 2 (random)
solver2 = Schnakenberg(300000, u0, v0, 10, 0.1, 0.9, 4000, 300000)


#solver.solve()
solver2.solve()

#solver.plot()
solver2.plot()
plt.show() 
"""

"""Aminación"""

"""M = solver.getAnimation()

def update(i):
    matrice.set_array(M[i])

fig, ax = plt.subplots()
matrice = ax.matshow(np.ones((50, 50)))
plt.colorbar(matrice)

ani = animation.FuncAnimation(fig, update, frames=80000, interval=20)
#ani.save('rd1.mp4')
plt.show()"""





import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
from scipy import sparse
from scipy.sparse import linalg
from matplotlib import animation
import time


"""
Esta clase se encarga de resolver el modelo DIB.
Se hace uso de un método DF explícito con actualización matricial. 
"""

class DIB:
    def __init__(self, Nt, dt, tMax, u0, v0):
        """
        Nt: Tamaño partición temporal. (Cantidad de iteraciones)
        u0: numpy array (N x N), solución u en t = 0.
        v0: numpy array (N x N), solución v en t = 0.
        """
        # Cond iniciales.
        self.u0 = u0
        self.v0 = v0
        # Parámetros.
        self.Nt = Nt
        self.tMax = tMax
        self.N = len(u0)
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
        Calcula y guarda la solución en tiempo final.  
        """
        U = self.u0
        V = self.v0
        alpha = (self.N + 1)**2
        dt = 1 
        du = 1
        dv = 20
    
        # Matriz
        A =  self.matriz()
        # Cantidad máxima de iteraciones.
        T = self.tMax  
        for i in np.arange(1, T + 1):
            U = U + du * alpha * (np.dot(A, U) + np.dot(U, A)) + dt * (F * (1 - U) - U * V**2)
            V = V + dv * alpha * (np.dot(A, V) + np.dot(V, A)) + dt * (U * V ** 2 - (F + k) * V)
            #if i % 10 == 0:
            #    self.M.append(U)
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
        axs[0, 1].matshow(self.v0)
        axs[1, 0].matshow(self.U)
        axs[1, 1].matshow(self.V)
        plt.show()
        #axs[2, 0].matshow(self.suma)
        #axs[2, 1].matshow(self.suma)


"""Test"""
N = 200
u0 = np.ones((N, N))
u0[10:20,10:20] = 0 
v0 = 1 - u0
solver = DIB(400000, 100000, u0, v0, 2 * 10 **-5, 10**-5, 0.05, 0.063)
solver.solve()
solver.plot()

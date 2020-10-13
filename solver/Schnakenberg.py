import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 
# Para guardar los videos.
plt.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\jigna\\ffmpeg\\bin\\ffmpeg.exe'
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.animation as animation
import time

"""
Esta clase se encarga de resolver el modelo de Schnakenberg con diferencias finitas.
Se hace uso de un método explícito DF con actualización matricial. 
Asume que la grilla es de tamaño N x N dado por las matrices de condiciones iniciales. 
"""
np.random.seed(6)

class Schnakenberg:
    def __init__(self, Nt, dx, dt, u0, v0, d, a, b, gamma, tMax = 100000):
        """
        Nt: Tamaño partición temporal. (Cantidad de iteraciones)
        dx: paso espacial (ambas direcciones).
        dt: paso temporal.
        u0: numpy array (N x N), solución u en t = 0.
        v0: numpy array (N x N), solución v en t = 0.
        d: Constante del problema.
        a: Constante del problema.
        gamma: Constante del problema.
        tMax: cantidad máxima de iteraciones.
        """
        # Parámetros.
        self.Nt = Nt
        self.dx = dx
        self.dt = dt
        self.N = len(u0)
        # Cond iniciales.
        self.u0 = u0
        self.v0 = v0
        # Constantes del modelo.
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
    
    
    
    def matriz(self, flag=True):
        """
        Crea y retorna la matriz asociadas a la discretización de derivadas.
        Creará la matríz para discretizar las derivadas en y.
        Usar de la forma A * U + U * A^T
        flag: boolean, true para condiciones de borde periódicas y false para flujo nulo en el borde.
        """
        offset = [-1, 0, 1]
        up = np.ones(self.N - 1) 
        ppal = - 2 * np.ones(self.N) 
        down = up
        k = np.array([down, ppal, up])
        A = sparse.diags(k, offset)
        A = A.toarray()
        # Condiciones de borde periódicas en ambos extremos.
        if flag:
            A[0][self.N - 2] = 1
            A[self.N - 1][1] = 1
        # Flujo nulo.
        else:
            A[0][1] = 2
            A[self.N - 1][self.N - 2] = 2
        return A


    def solve(self, flag=True):
        """
        Calcula y guarda la solución en tiempo final.
        flag: boolean, true para condiciones de borde periódicas y false para flujo nulo en el borde.
        """
        U = self.U
        V = self.V
        alpha = self.dt / (self.dx**2)
        d = self.d
        a = self.a
        b = self.b
        gamma = self.gamma
        A =  self.matriz(flag)
        At = np.transpose(A)
        # Cantidad máxima de iteraciones.
        T = self.tMax 
        ti = time.time()
        for i in np.arange(1, T + 1):
            U = U + alpha * (np.dot(A, U) + np.dot(U, At)) + self.dt * gamma * (a - U + U**2 * V)
            V = V + d * alpha * (np.dot(A, V) + np.dot(V, At)) + self.dt * gamma * (b - U**2 * V)
            if i % 1000 == 0:
                print('avance: ' + str(100 * i / Nt) + ' %')
                self.M.append(U)
        tf = time.time()
        print('min: ', (tf - ti) / 60)
        self.U = U
        self.V = V

    def animate(self, save=False):
        """
        Crea un video.
        save: boolean, true para guardar expecificando la ruta más abajo.
        """
        fig = plt.figure()

        ims = []
        for matriz in self.M:
            img = plt.imshow(matriz, animated=True)
            plt.title(r'$u$ con $\gamma=$' + str(self.gamma), y = 1, fontname='serif')
            plt.axis('off')
            ims.append([img])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
        if save:
            f = r"c://Users/jigna/Desktop/schnakenberg-"+ str(self.gamma) +".mp4" 
            writervideo = animation.FFMpegWriter(fps=10) 
            ani.save(f, writer=writervideo)
        plt.show()


    def plot(self):
        """
        Realiza un plot
        """
        f, axs = plt.subplots(1,2)

        axs[0].matshow(self.U)
        axs[0].set_title(r'$u$ con $\gamma=$' + str(self.gamma), y = 1, fontname='serif')

        axs[1].matshow(self.V)
        axs[1].set_title(r'$v$ con $\gamma=$' + str(self.gamma), y = 1, fontname='serif')

        plt.show()






"""Generar plots y videos"""

# Constructor -> def __init__(self, Nt, u0, v0, d, a, b, gamma, tMax = 100000)

N = 80
Nt = 3 * 10 ** 5
# Condición inicial aleatoria.

def aleatoria():
    u0 = np.random.random((N, N))
    v0 = np.random.random((N, N))

    aleatoria = Schnakenberg(Nt=Nt, dt=1/Nt, dx=1/N, u0=u0, v0=v0, d=10, a=0.1, b=0.9, gamma=7000, tMax=Nt)
    aleatoria.solve()
    aleatoria.animate()

# aleatoria()

# Condición inicial determinista
def determinista():

    u0 = np.zeros((N,N))
    u0[N//2-10:N//2+10, N//2-10:N//2+10] = 0.2
    v0 = np.zeros((N,N))
    v0[N//2-10:N//2+10, N//2-10:N//2+10] = 0.8

    determinista = Schnakenberg(Nt=Nt, dt=1/Nt, dx=1/N, u0=u0, v0=v0, d=10, a=0.1, b=0.9, gamma=3000, tMax=Nt)
    determinista.solve()
    determinista.plot()

determinista()




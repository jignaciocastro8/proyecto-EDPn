import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\jigna\\ffmpeg\\bin\\ffmpeg.exe'
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.animation as animation
import time


"""
Esta clase se encarga de resolver el modelo de Gray-Scott.
Hace uso de un método explícito, es inestable para variaciones de dt y dx.


"""
class GrayScott:
    def __init__(self, Nt, dx, dt, u0, v0, Du, Dv, F, k, tMax):
        """
        Nt: Tamaño partición temporal. (Cantidad de iteraciones)
        dx: paso espacial (ambas direcciones).
        dt: paso temporal.
        u0: numpy array (N x N), solución u en t = 0.
        v0: numpy array (N x N), solución v en t = 0.
        Du: coeficiente de difusión para u.
        Dv: coeficiente de difusión para v.
        F: Constante del problema.
        k: Constante del problema.
        tMax: Cantidad máxima de iteraciones
        """
        # Parámetros.
        self.Nt = Nt
        self.dx = dx
        self.dt = dt
        self.tMax = tMax
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
        # Soluciones.
        self.U = u0
        self.V = v0
        # Animación (solo de u)
        self.M = []
    
        

    def matrizLap(self, A):
        """
        Crea y retorna matriz para discretizar laplaceano.
        """
        n = len(A)

        # Nodo central.
        A0 = -4 * A

        # A1 nodo de la derecha. (columnas a la izq.)
        A1 = np.roll(A, (0,-1), (0,1))
        for i in range(n):
            A1[i, n - 1] = A[i, 1]
        
        # A2 nodo de la izquierda. (Columnas a la der.)
        A2 = np.roll(A, (0,+1), (0,1)) 
        for i in range(n):
            A2[i, 0] = A[i, n - 2]

        # A3 nodo de abajo (Filas para arriba)
        A3 = np.roll(A, (-1,0), (0,1)) 
        for j in range(n):
            A3[n - 1, j] = A[1, j]

        # A4 nodo de arriba (Filas para abajo)
        A4 = np.roll(A, (+1,0), (0,1))
        for j in range(n):
            A4[0, j] = A[n - 2, j]


        return A0 + A1 + A2 + A3 + A4
    def update(self, U, V):
        """
        Realiza un update a las matrices según la edp.
        SI SE CAMBIA A ALGO DE LA FORMA: U+= ... return U, NO FUNCIONA.
        """
        alpha = self.dt / (self.dx ** 2)
        U0 = U + alpha * self.Du * self.matrizLap(U) + (self.F * (1-U) - U * V**2) * self.dt
        V0 = V + alpha * self.Dv * self.matrizLap(V) + (U * V**2 - (self.k + self.F)*V) * self.dt
    
        return U0, V0


    def solve(self):
        """
        Resuelve iterativamente.
        """
        U = self.U 
        V = self.V
        A = []
        ti = time.time()
        for i in range(self.tMax):
            if i % 100 == 0:
                print('iter n°: ', i)
                A.append(U)
            U, V = self.update(U, V)
            
        tf = time.time()
        print('tiempo (min):', (tf - ti) / 60)
        self.M = A
        self.U = U
        self.V = V
        #return A

    def animate(self, save=False):
        """
        Retorna un arreglo con soluciones de u en el tiempo.
        """
        fig = plt.figure()

        ims = []
        for matriz in self.M:
            img = plt.imshow(matriz, animated=True)
            ims.append([img])   

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
        if save:
            f = r"c://Users/jigna/Desktop/grayScott.mp4" 
            writervideo = animation.FFMpegWriter(fps=10) 
            ani.save(f, writer=writervideo)
        plt.show()
    

    def plot(self):
        """
        Realiza un plot
        """

        f, axs = plt.subplots(1,2)

        axs[0].matshow(self.U)
        axs[0].set_title(r'$u$', y = 1, fontname='serif')

        axs[1].matshow(self.V)
        axs[1].set_title(r'$v$', y = 1, fontname='serif')







"""Test"""

# Constructor -> def __init__(self, Nt, dx, dt, u0, v0, Du, Dv, F, k, tMax = Nt)
N = 200
N2 = N//2
radius = r = int(N/10.0)
u0 = 0.8 * np.ones((N,N)) + 0.2 * np.random.random((N,N))
v0 = 0.2 * np.random.random((N,N))
u0[N2-r:N2+r, N2-r:N2+r] = 0.50
v0[N2-r:N2+r, N2-r:N2+r] = 0.25
Nt = 4 * 10 ** 3
solver = GrayScott(Nt=Nt, dx=1, dt=1, u0=u0, v0=v0, Du=0.16, Dv=0.08, F=0.06, k=0.062, tMax=10000)


solver.solve()
solver.animate()
"""fig = plt.figure()

ims = []
for matriz in M:
    img = plt.imshow(matriz, animated=True)
    ims.append([img])


ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)

f = r"c://Users/jigna/Desktop/grayScott.mp4" 
writervideo = animation.FFMpegWriter(fps=10) 
ani.save(f, writer=writervideo)

plt.show()"""



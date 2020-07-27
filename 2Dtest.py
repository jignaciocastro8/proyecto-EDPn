#from RDsolver2D import *
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# Mesh
dx = 0.1
x = np.arange(dx, 5, dx)
y = np.arange(dx, 5, dx)
xx, yy = np.meshgrid(x, y, sparse=False, indexing='ij')


# Función indicatríz de un sub-cuadradito en el cuadradro [0, 1]x[0, 1]
def f0(x, y):
    return (0.4 <= x <= 0.6) and (0.4 <= y <= 0.6) 

# Gaussiana
g0 = lambda x,y : np.exp(-((x - 0.5)**2 + (y - 0.5)**2))
    


# Constructor : def __init__(self, Nt, u0, v0, d, a, b, gamma)

"""N = 100

A1 = np.zeros((N, N))
A1[5:25, 5:25] = 1
A1[30:40, 30:40] = 1
B1 = 1 - A1

u0 = np.zeros((N, N))
u0[20:30, 20:30] = 1
v0 = 1 - u0

solver = RDsolver2D(400000, A1, B1, 10, 0.1, 0.9, 1000)
solver2 = RDsolver2D(400000, u0, v0, 10, 0.1, 0.9, 1000)
ti = time.time()
solver.solve()
solver2.solve()
tf = time.time()
print(tf - ti)
solver.plot()
solver2.plot()
plt.show() """  


plt.matshow(np.random.random((50, 50)), cmap='nipy_spectral')
plt.show()


    
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
    

def update(i):
    matrice.set_array(np.random.random((50, 50)))

fig, ax = plt.subplots()
matrice = ax.matshow(np.random.random((50, 50)))
plt.colorbar(matrice)

ani = animation.FuncAnimation(fig, update, frames=19, interval=500)
plt.show()
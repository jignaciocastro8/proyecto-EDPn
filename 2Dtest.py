from RDsolver2D import *
import time
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
    


def get_initial_configuration(N, random_influence=0.01):
    """
    Initialize a concentration configuration. N is the side length
    of the (N x N)-sized grid.
    `random_influence` describes how much noise is added.
    """
    
    # We start with a configuration where on every grid cell 
    # there's a lot of chemical A, so the concentration is high
    A = (1-random_influence) * np.ones((N,N)) + random_influence * np.random.random((N,N))
    
    # Let's assume there's only a bit of B everywhere
    B = random_influence * np.random.random((N,N))
    
    # Now let's add a disturbance in the center
    N2 = N//2
    N4 = N//4
    r = int(N/10)
    
    #A[N2-N4-r:N2-N4+r, N2-N4-r:N2-N4+r] = 0.50
    A[N2-r:N2+r, N2-r:N2+r] = 0.50
    B[N2-r:N2+r, N2-r:N2+r] = 0.25
    
    return A, B

# Constructor : def __init__(self, Nt, u0, v0, d, a, b, gamma)

N = 50
A, B = get_initial_configuration(N, 0)
A1 = np.zeros((N, N))
A1[5:25, 5:25] = 1
A1[30:40, 30:40] = 1
B1 = 1 - A1

# Funciones en t = 0.

u0 = np.random.random((N, N))
v0 = np.random.random((N, N)) 
a = 1
solver = RDsolver2D(400000, A1, B1, 10, 0.1, 0.9, 1000)
solver2 = RDsolver2D(400000, A, B, 10, 0.1, 0.9, 1000)
ti = time.time()
solver.solve()
solver2.solve()
tf = time.time()
print(tf - ti)
solver.plot()
solver2.plot()
plt.show()   
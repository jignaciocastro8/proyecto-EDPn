from Schnakenberg1D import *


# Constructor: def __init__(self, Nt, dx, dt, u0, v0, d, a, b, gamma)


N = 100
dx = 0.1

Nt = 10 ** 8
dt = 10 ** -8


a = 0.126779
b = 0.792366

u0 = np.ones(N) * (a + b) + np.random.random(N) * 10**-3
v0 = np.ones(N) * (b / (a + b)**2) + np.random.random(N) * 10**-3
solver = Schnakenberg1D(Nt, dx, dt, u0, v0, 10, a, b, 2000)
solver.solve()
solver.plot()   
 

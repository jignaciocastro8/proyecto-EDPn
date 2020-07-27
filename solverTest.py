from RDsolver import *

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

# En 2 dimensiones.




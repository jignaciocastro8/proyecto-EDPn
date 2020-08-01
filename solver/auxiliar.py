import numpy as np
import matplotlib.pyplot as plt
from Schnakenberg import *

N = 80

u0 = np.zeros((N, N))
v0 = np.zeros((N, N))

u0[10:20, 10:20] = 0.5
u0[30:40, 30:40] = 0.5

v0[40:50, 40:50] = 0.5
v0[60:70, 60:70] = 0.5

f, ax = plt.subplots(1,2)
ax[0].matshow(u0, cmap='Blues')
ax[0].set_title(r'$u(t=0)$', y = 1, fontname='serif')
ax[1].matshow(v0, cmap='Blues')
ax[1].set_title(r'$v(t=0)$', y = 1, fontname='serif')

#plt.show()
# Condición inicial particular.
solver = Schnakenberg(300000, u0, v0, 10, 0.1, 0.9, 3000, 300000)
solver.solve()
solver.plot()
plt.show()

arr = np.array([210000])
meanU = []
meanV = []
for Nt in arr:
    solver.setNt(Nt)
    solver.setTmax(Nt)
    solver.solve()
    meanU.append(solver.getMean()[0])
    meanV.append(solver.getMean()[1])

plt.title('Promedio')
plt.xlabel(r'$N_t$')
plt.plot(meanU, '*')
plt.plot(meanV, '*')
plt.grid()
plt.show()
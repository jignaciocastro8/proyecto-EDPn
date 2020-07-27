import numpy as np #Biblioteca basica para c´alculo cient´ıfico
import scipy as sp #Biblioteca avanzada para matrices dispersas
import matplotlib.pyplot as plt #Biblioteca para graficar
from scipy import sparse
from scipy.sparse import linalg
from matplotlib.animation import FuncAnimation
from matplotlib import cm

#d constante del problema
#dt paso del tiempo
#NT total de discretizaciones en el tiempo
#1/N tamaño de discretizacion en el espacio
#D rango espacial
#u0 v0 son las condiciones iniciales
#a es una funcion del problema
def calcula_sol(d,dt,NT,N,D,u0,v0,a):
    dx = D/N;
    alfa = dt/(dx)**2;
    t = 1-dt-(2*d*alfa);
    ap = 1-2*alfa;
    e1 = np.ones(N);
    e2 = np.ones(N-1);
    k = np.array([alfa*d*e2,t*e1,alfa*d*e2]);
    kp = np.array([alfa*e2,ap*e1,alfa*e2]);
    offset = [-1,0,1];
    A = sp.sparse.diags(k,offset,format='csc');
    B = sp.sparse.diags(kp,offset,format='csc')
    A[0,0] = 1-dt-(d*alfa);
    A[N-1,0] = d*alfa;
    Mu = np.zeros((N+1,1));
    for i in range(N+1):
        Mu[i,0] = u0(i*dx)
    Mv = np.zeros((N+1,1));
    for j in range(N+1):
        Mv[j,0] = v0
    ui0 = Mu[:,0]
    vi0 = Mv[:,0]
    for i in range(1,NT+1):
        u0v0 = np.ones(N);
        for j in range(N):
            u0v0[j] = ui0[j+1]*vi0[j+1];
        a0 = np.ones(N);
        for j in range(N):
            a0[j] = a(ui0[j+1],vi0[j+1])*ui0[j+1];
        u = A*ui0[1:N+1] + dt*u0v0 - dt*a0;
        v = B*vi0[1:N+1] - dt*u0v0;
        h = u[0];
        u = np.concatenate((u,[h]));
        h2 = v[0];
        v = np.concatenate((v,[h2]));
        Mu = np.insert(Mu,Mu.shape[1],u,1);
        Mv = np.insert(Mv,Mv.shape[1],v,1);
        ui0 = Mu[:,i];
        vi0 = Mv[:,i];
    return Mu,Mv

d=10;
dt = 1/100;
NT=500;
N=1000;
D=20;
u0 = lambda x: x;
v0 = 5;
a = lambda x,y: x+y;

sol = calcula_sol(d,dt,NT,N,D,u0,v0,a)

tt = np.arange(0,NT*dt+dt,dt);
xx = np.arange(0,D+D/N,D/N);
X,Y = np.meshgrid(tt,xx);
fig,ax = plt.subplots()
im = ax.contourf(X,Y,sol[0],100,cmap='jet');
fig.colorbar(im,ax=ax);
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:34:24 2020

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt
def RK2(f,u,times,subdiv = 1):
     uout = np.zeros((len(times),)+u.shape)
     uout[0] = u;
     for k in range(len(times)-1):
         t = times[k]
         h = (times[k+1]-times[k])/subdiv
         for j in range(subdiv):
            k1 = f(u,t)*h
            k2 = f(u+0.5*k1, t+0.5*h)*h
            u, t = u+k2, t+h
         uout[k+1]=u
     return uout

def plotphase(A,B,C,D,E):
     def derivs(u,t): y,z = u; return np.array([ z, -A*y**3 + B*y - C*z + D*np.cos(E*t) ])
     N=60
     u0 = np.array([0.0, 0.0])
     t  = np.arange(0,300,2*np.pi/N); 
     u  = RK2(derivs, u0, t, subdiv = 10)
     plt.plot(u[:-2*N,0],u[:-2*N,1],'.--y', u[-2*N:,0],u[-2*N:,1], '.-b', lw=0.5, ms=2);
     plt.plot(u[::N,0],u[::N,1],'rs', ms=4); plt.grid(); plt.show()
     return u

l = plotphase(1.0, 5.0, 0.02, 8.0, 0.5)

import nolds
qr = nolds.lyap_r(l[:,0])
qe = nolds.lyap_e(l[:,0])

import nolitsa.lyapunov as qn
#qn = nolitsa.lyapunov
qww = qn.mle(l)
maximum= np.amax(qww)

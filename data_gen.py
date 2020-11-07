# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:51:52 2020

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt
import nolitsa.lyapunov as qn
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
     qww = qn.mle(u)
     return np.amax(qww)

a = np.linspace(1,50,num = 10)
b = np.linspace(1,250, num = 10)
c = np.linspace(0.02, 25, num=10)
d = np.linspace(1,125, num = 10)
e = np.linspace(0.5, 60, num = 10)
z=0
n = np.zeros(99999)
for i in a:
    for j in b:
        for k in c:
            for l in d:
                for m in e:
                    n[z] = plotphase(i,j,k,l,m)
                    z=z+1
np.savetxt("data.txt", n)                    

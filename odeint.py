# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:58:05 2020

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def f(y, t, params):
    theta, omega = y      # unpack current values of y
    delta, alpha, beta, gamma, Omega = params  # unpack parameters
    derivs = [omega,      # list of dy/dt=f functions
             -delta*omega-alpha*theta-beta*np.power(theta,3)+gamma*np.cos(Omega*t)]
    return derivs

# Parameters
alpha = 1.0          # linear stiffness
beta = 5.0          # non-linearity in restoring force
gamma = 0.02     # amplitude of driving force
delta = 8.    # Amount of damping
Omega = 0.5     # Angular freq

# Initial values
theta0 = 0.0     # initial angular displacement
omega0 = 0.0     # initial angular velocity

# Bundle parameters for ODE solver
params = [alpha, beta, gamma, delta, Omega]

# Bundle initial conditions for ODE solver
y0 = [theta0, omega0]

# Make time array for solution
tStop = 1000.
tInc = 0.05
t = np.arange(0., tStop, tInc)

# Call the ODE solver
psoln = odeint(f, y0, t, args=(params,))

# Plot results
fig = plt.figure(1, figsize=(8,8))

# Plot theta as a function of time
ax1 = fig.add_subplot(311)
ax1.plot(t, psoln[:,0])
ax1.set_xlabel('time')
ax1.set_ylabel('theta')

# Plot omega as a function of time
ax2 = fig.add_subplot(312)
ax2.plot(t, psoln[:,1])
ax2.set_xlabel('time')
ax2.set_ylabel('omega')

# Plot omega vs theta
ax3 = fig.add_subplot(313)
twopi = 2.0*np.pi
ax3.plot(psoln[:,0]%twopi, psoln[:,1], '.', ms=1)
ax3.set_xlabel('theta')
ax3.set_ylabel('omega')
ax3.set_xlim(0., twopi)

plt.tight_layout()
plt.show()


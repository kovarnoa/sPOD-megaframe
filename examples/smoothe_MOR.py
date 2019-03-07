# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:11:12 2018

@author: philipp krah, jiahan wang
"""

###############################################################################
# IMPORTED MODULES
###############################################################################
import sys
sys.path.append('../lib')
from numpy import *
from numpy.linalg import svd
import numpy as np
from numpy import exp, mod,meshgrid
import matplotlib.pyplot as plt
###############################################################################

N = np.asarray([100, 101])
L = np.asarray([1, 1])
delta = 0.02    # peak at the midpoint of the soft field
lambd = 0.01    # thickness of the step
R = 0.25        # radius of the cylinder
Nmodes=10
x=[]
x.append(np.linspace(0,L[0],N[0]))
x.append(np.linspace(0,L[1],N[1]))

[X,Y] = np.meshgrid(x[1],x[0])

n_snapshots = 25;

phis = np.zeros([N[0] ,N[1], n_snapshots])
qs = np.zeros([N[0] ,N[1], n_snapshots])

f = lambda x,l : ((np.tanh(x/l) + 1 ) * 0.5)

for k in range(n_snapshots):
    z = np.exp(1j*2*np.pi *k/n_snapshots)
    x0= 0.5 + R*np.real(z)
    y0= 0.5 + R*np.imag(z)
    
    phi = sqrt((X-x0)**2 + (Y-y0)**2 + delta**2) -0.15 -delta;
    q   = f(phi,lambd); 

    phis[:,:,k] = phi ;    # store distance function 
    qs[:,:,k] = q   ;    # store field 
    
    if 0:
        plt.subplot(131)
        plt.pcolor(X,Y,phi)
        plt.title("soft")
        plt.ylabel("$x$")
        plt.xlabel("$y$")

       
        
        plt.subplot(132)
        plt.title("sharp")
        plt.pcolor(X,Y,q)
        plt.ylabel("$x$")
        plt.xlabel("$y$")
        plt.pause(0.01)
        plt.colorbar()
        
# %%
[Uphi,Sphi,Vphi] = svd(reshape(phis,[-1,n_snapshots]), full_matrices=True)
Sphi = Sphi[:Nmodes]
Uphi = Uphi[:, :Nmodes]
Vphi = Vphi[:Nmodes, :]

[Uq,Sq,Vq] = svd(reshape(qs,[-1,n_snapshots]),full_matrices=True)
Sq = Sq[:Nmodes]
Uq = Uq[:, :Nmodes]
Vq = Vq[:Nmodes, :]

        
# %%
ntime=2
phi_appr=np.reshape(np.dot(Uphi * Sphi, Vphi), [N[0],N[1],n_snapshots])
fphi=f(phi_appr,lambd)
q_appr=np.reshape(np.dot(Uq * Sq, Vq), [N[0],N[1],n_snapshots])

plt.subplot(231)
plt.pcolor(X,Y,fphi[:,:,ntime])
plt.title("$f(\\tilde{\phi})$")
plt.ylabel("$x$")
plt.xlabel("$y$")
plt.subplot(234)
plt.pcolor(X,Y,fphi[:,:,ntime]-qs[:,:,ntime])
plt.title("$\\tilde{f}(\phi)-q$")
plt.ylabel("$x$")
plt.xlabel("$y$")

plt.subplot(232)
plt.title("q svd")
plt.pcolor(X,Y,q_appr[:,:,ntime])
plt.ylabel("$x$")
plt.xlabel("$y$")
plt.pause(0.01)

plt.subplot(235)
plt.pcolor(X,Y,q_appr[:,:,ntime]-qs[:,:,ntime])
plt.title("$\\tilde{q}-q$")
plt.ylabel("$x$")
plt.xlabel("$y$")


plt.subplot(133)
plt.title("q")
plt.pcolor(X,Y,qs[:,:,ntime])
plt.ylabel("$x$")
plt.xlabel("$y$")
plt.colorbar()
plt.pause(0.01)




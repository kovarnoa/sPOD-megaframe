#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOVING CYLINDERS VORTEX STREET

Created on Mon May  3 22:18:56 2021

@author: miriam
"""

###############################################################################
# IMPORTED MODULES
###############################################################################
import sys
sys.path.append('../lib')
import numpy as np
from numpy import exp, mod,meshgrid,pi,sin,size
import matplotlib.pyplot as plt
from sPOD_tools import frame, sPOD_distribute_residual
from transforms import transforms
from scipy.io import loadmat

from sympy.physics.vector import curl
###############################################################################

##########################################
#%% Define your DATA:
##########################################
plt.close("all")
ux = loadmat('ux.mat')
ux = ux['data']
uy = loadmat('uy.mat')
uy = uy['data']

Ngrid = [ux.shape[2], ux.shape[3]]  # number of grid points in x
Nt = ux.shape[1]                      # Number of time intervalls
Nvar = ux.shape[0]                    # Number of variables
nmodes = 1                              # reduction of singular values

data_shape = [*Ngrid,Nvar,Nt]

data = ux
#data = np.zeros(data_shape)
#for tau in range(0,Nt):
#    data[0,tau,:,:] = curl(np.squeeze(ux[0,tau,:,:]),np.squeeze(uy[0,tau,:,:]))

               # size of time intervall
T = 1000.                # total time
L = np.asarray([1, 1])   # total domain size
x,y = (np.linspace(0, L[i], Ngrid[i]) for i in range(2))
time = np.linspace(0, T, Nt)
dx,dy = (x[1]-x[0],y[1]-y[0])
dt = time[1]-time[0]
c = dx/dt
[Y,X] = meshgrid(y,x)

q = np.zeros(data_shape)
for it,t in enumerate(time):
    q[:,:,0,it] = np.array(data[0,it,:,:]).T

shift1 = np.zeros([2,Nt])
shift2 = np.zeros([2,Nt])
shift1[0,:] = 0 * time                      # frame 1, shift in x
shift1[1,:] = 0 * time                      # frame 1, shift in y
shift2[0,:] = 0 * time                      # frame 2, shift in x
shift2[1,:] = -0.25*np.sin(2*pi*0.001*time) # frame 2, shift in y

# %% Create Trafo

shift_trafo_1 = transforms(data_shape,L, shifts = shift1, dx = [dx,dy] )
shift_trafo_2 = transforms(data_shape,L, shifts = shift2, dx = [dx,dy] )
qshift1 = shift_trafo_1.apply(q)
qshift2 = shift_trafo_2.apply(q)
qshiftreverse = shift_trafo_2.reverse(shift_trafo_2.apply(q))
res = q-qshiftreverse
err = np.linalg.norm(np.reshape(res,-1))/np.linalg.norm(np.reshape(q,-1))
print("err =  %4.4e "% err)
plt.pcolormesh(X,Y,q[...,0,0]-qshiftreverse[...,0,0])
plt.colorbar()

    
# %% Run shifted POD
transforms = [shift_trafo_1, shift_trafo_2]
qframes, qtilde = sPOD_distribute_residual(q, transforms, nmodes=2, eps=1e-4, Niter=10, visualize=True)

plt.pcolormesh(X,Y,q[:,:,0,5]-qtilde[:,:,0,5])
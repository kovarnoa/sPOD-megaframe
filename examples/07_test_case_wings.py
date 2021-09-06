#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 08:31:20 2021

@author: miri
"""

###############################################################################
# IMPORTED MODULES
###############################################################################
import sys
sys.path.append('../lib')
import numpy as np
from numpy import exp, mod,meshgrid,pi,sin,size
import matplotlib.pyplot as plt
from sPOD_tools import frame, shifted_POD,shifted_rPCA
from transforms import transforms
from farge_colormaps import farge_colormap_multi
from scipy.io import loadmat
###############################################################################
cm = farge_colormap_multi()
##########################################
#%% Define your DATA:
##########################################
plt.close("all")
data = loadmat('ALL.mat')
data = data['data']
data = data[:,:,352:1056,0:704]

Ngrid = [data.shape[2], data.shape[3]]  # number of grid points in x and y
Nt = data.shape[1]                      # Number of time intervalls
Nvar = data.shape[0]                    # Number of variables
nmodes = 3                              # reduction of singular values

data_shape = [*Ngrid,Nvar,Nt]
               # size of time intervall
T = 3                                   # total time
L = np.asarray([1, 1])   # total domain size
x0 = L*0.5               # starting point of the gauss puls
R = 0.1 * min(L)         # Radius of cylinder
x,y = (np.linspace(0, L[i], Ngrid[i]) for i in range(2))
time = np.linspace(0, T, Nt)
dx,dy = (x[1]-x[0],y[1]-y[0])
dt = time[1]-time[0]
c = dx/dt
[Y,X] = meshgrid(y,x)

rotation1   = np.zeros([Nt])
rotation1   = pi/6 * np.cos(2*pi*time) + pi/3 * np.sin(2*pi*time)    # frame 1
shift1      = np.zeros([2,Nt])
rotation2   = np.zeros([Nt])
rotation2   = pi/6 * np.cos(2*pi*time) + pi/3 * np.sin(2*pi*time)   # frame 2
rotation3   = np.zeros([Nt])

q = np.zeros(data_shape)
#for nvar in range(Nvar):
for it,t in enumerate(time):
    q[:,:,0,it] = np.array(data[0,it,:,:]).T
    
    shift1[0,it] = 0
    shift1[1,it] = 0.5 * np.cos(2*pi*t)
   

# %% Create Trafo
#trafo_1 = transforms(data_shape,L, shifts = shift1, dx = [dx2,dy2], use_scipy_transform=False )
trafo_1 = transforms(data_shape,L, trafo_type="shiftRot", shifts = shift1, dx = [dx,dy], rotations = rotation1, rotation_center=[0.4*L[0],0.5*L[1]], use_scipy_transform=False)
trafo_2 = transforms(data_shape,L, trafo_type="rotation", dx = [dx,dy], rotations = rotation2, rotation_center=[0.6*L[0],0.5*L[1]], use_scipy_transform=False )

qshift1 = trafo_1.reverse(q)
qshift2 = trafo_2.reverse(q)
qshiftreverse = trafo_1.apply(trafo_1.reverse(q))
res = q-qshiftreverse
err = np.linalg.norm(np.reshape(res,-1))/np.linalg.norm(np.reshape(q,-1))
print("err =  %4.4e "% err)
plt.pcolormesh(X,Y,q[...,0,0]-qshiftreverse[...,0,0],cmap = cm)
plt.colorbar()
# %% Test Trafo

# figs,axs = plt.subplots(3,1,sharex=True)
# axs[0].pcolormesh(X,Y,qshift1[...,0,0])
# axs[1].pcolormesh(X,Y,qshift2[...,0,0])
# axs[2].pcolormesh(X,Y,q[...,0,0]-qshiftreverse[...,0,0])
# axs[0].set_title(r"$q^1$ Frame 1")
# axs[1].set_title(r"$q^2$ Frame 2")
# axs[2].set_title(r"$q - T^s_1 q^1 + T^s_2 q^2$ Data")
# for it in range(Nt):
#     axs[0].pcolormesh(X,Y,qshift1[...,0,it])
#     axs[1].pcolormesh(X,Y,qshift2[...,0,it])
#     axs[2].pcolormesh(X,Y,q[...,0,it]-qshiftreverse[...,0,it])
#     plt.show()
#     plt.pause(0.001)
    
# %% Run shifted POD
transforms = [trafo_1, trafo_2]
qmat = np.reshape(q,[-1,Nt])
#qframes, q = shifted_POD(q, transforms, nmodes=2, eps=1e-4, Niter=20, visualize=True)
mu = np.prod(np.size(qmat)) / (4 * np.sum(np.abs(qmat)))*0.1
ret = shifted_rPCA(qmat, transforms, eps=1e-4, Niter=50, visualize=True, mu = mu)
qframes, qtilde = ret.frames, ret.data_approx
qtilde = np.reshape(qtilde,data_shape)
plt.pcolormesh(X,Y,q[...,0,10]-qtilde[...,0,10])

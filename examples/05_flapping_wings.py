#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLAPPING WINGS 

Created on Tue May  4 23:50:42 2021

@author: Miriam
"""

###############################################################################
# IMPORTED MODULES
###############################################################################
import sys
sys.path.append('../lib')
import numpy as np
from numpy import exp, mod,meshgrid,pi,sin,size
import matplotlib.pyplot as plt
from sPOD_tools import frame, shifted_POD
from transforms import transforms
from scipy.io import loadmat
###############################################################################

##########################################
#%% Define your DATA:
##########################################
plt.close("all")
data = loadmat('ALL2.mat')
data = data['data']

Ngrid = [data.shape[2], data.shape[3]]  # number of grid points in x
Nt = data.shape[1]                      # Number of time intervalls
Nvar = 1# data.shape[0]                    # Number of variables
nmodes = 3                              # reduction of singular values

data_shape = [*Ngrid,Nvar,Nt]
               # size of time intervall
freq    = 0.1
T       = 1/freq*4       # total time
L = np.asarray([1, 1])   # total domain size
x,y = (np.linspace(0, L[i], Ngrid[i]) for i in range(2))
time = np.linspace(0, T, Nt)
dx,dy = (x[1]-x[0],y[1]-y[0])
dt = time[1]-time[0]
c = dx/dt
[Y,X] = meshgrid(y,x)

q = np.zeros(data_shape)
#for nvar in range(Nvar):
for it,t in enumerate(time):
    q[:,:,0,it] = np.array(data[4,it,:,:]).T

rotation1 = np.zeros([Nt])
rotation2 = np.zeros([Nt])
rotation3 = np.zeros([Nt])
rotation1 = pi/4 * np.cos(2*pi*freq*time)    # frame 1
rotation2 = -pi/4 * np.cos(2*pi*freq*time)   # frame 2

# %% Create Trafo

rotation_trafo_1 = transforms(data_shape,L,trafo_type="rotation",dx=[dx,dy],rotations=rotation1,rotation_center=[0*L[0],0*L[1]])
rotation_trafo_2 = transforms(data_shape,L,trafo_type="rotation",dx=[dx,dy],rotations=rotation2,rotation_center=[0*L[0],0*L[1]])
rotation_trafo_3 = transforms(data_shape,L,trafo_type="rotation",dx=[dx,dy],rotations=rotation3,rotation_center=[0*L[0],0*L[1]])
qshift1 = rotation_trafo_1.reverse(q)
qshift2 = rotation_trafo_2.reverse(q)
qshift3 = rotation_trafo_3.reverse(q)
qshiftreverse = rotation_trafo_1.reverse(rotation_trafo_1.apply(q))
res = q-qshiftreverse
err = np.linalg.norm(np.reshape(res,-1))/np.linalg.norm(np.reshape(q,-1))
print("err =  %4.4e "% err)
plt.pcolormesh(X,Y,q[...,0,0]-qshiftreverse[...,0,0])
plt.colorbar()

    
# %% Run shifted POD
transforms = [rotation_trafo_1, rotation_trafo_2, rotation_trafo_3]
ret = shifted_POD(q, transforms, nmodes=3, eps=1e-4, Niter=10, visualize=True)
qframes, qtilde , rel_err_list = ret.frames, ret.data_approx, ret.rel_err_hist
plt.pcolormesh(X,Y,q[:,:,0,5]-qtilde[:,:,0,5])

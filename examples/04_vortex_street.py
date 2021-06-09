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
from transforms import transforms
from sPOD_tools import frame, shifted_POD, shifted_rPCA, build_all_frames
from scipy.io import loadmat
from numpy.fft import fft2,ifft2
from farge_colormaps import farge_colormap_multi
###############################################################################
cm = farge_colormap_multi()
#from sympy.physics.vector import curl
###############################################################################

##########################################
#%% Define your DATA:
##########################################
plt.close("all")
dir = "/home/phil/develop/two_cylinders/"
data = loadmat(dir+'ux.mat')
ux = data['data']
data= loadmat(dir+'uy.mat')
uy = data['data']



Ngrid = [ux.shape[2], ux.shape[3]]  # number of grid points in x
Nt = ux.shape[1]                      # Number of time intervalls
Nvar = 2                # Number of variables
nmodes = 1                              # reduction of singular values

data_shape = [*Ngrid,Nvar,Nt]

#data = np.zeros(data_shape)
#for tau in range(0,Nt):
#    data[0,tau,:,:] = curl(np.squeeze(ux[0,tau,:,:]),np.squeeze(uy[0,tau,:,:]))

               # size of time intervall
T = 1000.                # total time
L = np.asarray([1, 1])   # total domain size
x,y = (np.linspace(0, L[i], Ngrid[i]) for i in range(2))
time = np.linspace(0, T, Nt)
dX = (x[1]-x[0],y[1]-y[0])
dt = time[1]-time[0]
[Y,X] = meshgrid(y,x)
dim = 2
#
# K = [np.fft.fftfreq(Ngrid[k], d=dX[k]) for k in range(dim)]
# [kx, ky] = meshgrid(K[0], K[1])
# uxhat = fft2(ux)
# uyhat = fft2(uy)
# dux_dy = np.real(ifft2(ky * uxhat * (1j)))
# duy_dx = np.real(ifft2(kx * uyhat * (1j)))
# vort = duy_dx - dux_dy

q = np.zeros(data_shape)
ux=np.transpose(ux,[2,3,0,1])
uy=np.transpose(uy,[2,3,0,1])
q[:,:,0,:] = np.squeeze(ux)
q[:,:,1,:] = np.squeeze(uy)
shift1 = np.zeros([2,Nt])
shift2 = np.zeros([2,Nt])
shift1[0,:] = 0 * time                      # frame 1, shift in x
shift1[1,:] = 0 * time                      # frame 1, shift in y
shift2[0,:] = 0 * time                      # frame 2, shift in x
shift2[1,:] = -0.25*np.sin(2*pi*0.001*time) # frame 2, shift in y

# %% Create Trafo

shift_trafo_1 = transforms(data_shape,L, shifts = shift1, dx = dX, use_scipy_transform=False )
shift_trafo_2 = transforms(data_shape,L, shifts = shift2, dx = dX, use_scipy_transform=False )
qshift1 = shift_trafo_1.reverse(q)
qshift2 = shift_trafo_2.reverse(q)
qshiftreverse = shift_trafo_2.apply(shift_trafo_2.reverse(q))
res = q-qshiftreverse
err = np.linalg.norm(np.reshape(res,-1))/np.linalg.norm(np.reshape(q,-1))
print("err =  %4.4e "% err)
plt.pcolormesh(X,Y,q[...,0,0]-qshiftreverse[...,0,0])
plt.colorbar()

    
# %% Run shifted POD
trafos = [shift_trafo_1, shift_trafo_2]
qmat = np.reshape(q, [-1, Nt])

ret = shifted_rPCA(qmat, trafos, nmodes_max = np.max(nmodes)+10, eps=1e-10, Niter=50, visualize=True, use_rSVD=True)
qframes, qtilde , rel_err_list = ret.frames, ret.data_approx, ret.rel_err_hist
plt.pcolormesh(X,Y,q[:,:,0,5]-qtilde[:,:,0,5])
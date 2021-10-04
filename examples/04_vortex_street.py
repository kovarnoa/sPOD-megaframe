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
data = loadmat(dir+'vorx_19.mat')
data = data['data']


Nt = data.shape[1]                      # Number of time intervalls
Nvar = 1# data.shape[0]                    # Number of variables
nmodes = [6,4]                              # reduction of singular values
frac = 4


Ngrid = [data.shape[2]//frac, data.shape[3]//frac]  # number of grid points in x
data_shape = [*Ngrid,Nvar,Nt]
               # size of time intervall
freq    = 0.01
T       = 100       # total time
L = np.asarray([1, 1])   # total domain size
x,y = (np.linspace(0, L[i], Ngrid[i]) for i in range(2))
time = np.linspace(0, T, Nt)
dX = (x[1]-x[0],y[1]-y[0])
dt = time[1]-time[0]
[Y,X] = meshgrid(y,x)

q = np.zeros(data_shape)
#for nvar in range(Nvar):
for it,t in enumerate(time):
    q[:,:,0,it] = np.array(data[0,it,::frac,::frac]).T

data = q
# %%data = np.zeros(data_shape)
#for tau in range(0,Nt):
#    data[0,tau,:,:] = curl(np.squeeze(ux[0,tau,:,:]),np.squeeze(uy[0,tau,:,:]))

               # size of time intervall

shift1 = np.zeros([2,Nt])
shift2 = np.zeros([2,Nt])
shift1[0,:] = 0 * time                      # frame 1, shift in x
shift1[1,:] = 0 * time                      # frame 1, shift in y
shift2[0,:] = 0 * time                      # frame 2, shift in x
shift2[1,:] = -0.25*np.sin(2*pi*freq*time) # frame 2, shift in y

# %% Create Trafo

shift_trafo_1 = transforms(data_shape,L, shifts = shift1,trafo_type="identity", dx = dX, use_scipy_transform=False )
shift_trafo_2 = transforms(data_shape,L, shifts = shift2, dx = dX, use_scipy_transform=True )
qshift1 = shift_trafo_1.reverse(q)
qshift2 = shift_trafo_2.reverse(q)
qshiftreverse = shift_trafo_2.apply(qshift2)
res = q-qshiftreverse
err = np.linalg.norm(np.reshape(res,-1))/np.linalg.norm(np.reshape(q,-1))
print("err =  %4.4e "% err)
plt.pcolormesh(X,Y,q[...,0,0]-qshiftreverse[...,0,0])
plt.colorbar()

# %% Run shifted POD
trafos = [shift_trafo_1, shift_trafo_2]
qmat = np.reshape(q, [-1, Nt])



#ret = shifted_rPCA(qmat, trafos, nmodes_max = np.max(nmodes)+100, eps=1e-10, Niter=50, visualize=True, use_rSVD=True,lambd=1000000)
NmodesMax = 30
rel_err_matrix = np.ones([30]*len(trafos))
for r1 in range(0,NmodesMax):
    for r2 in range(0,NmodesMax):
        print("mode combi: [", r1,r2,"]\n")
        ret = shifted_POD(qmat, trafos, nmodes=[r1,r2], eps=1e-4, Niter=40, use_rSVD=True)
        print("\n\nmodes: [", r1, r2, "] error = %4.4e \n\n"%ret.rel_err_hist[-1])
        rel_err_matrix[r1,r2] = ret.rel_err_hist[-1]


np.savetxt('sPOD_error.txt', rel_err_matrix)

qframes, qtilde , rel_err_list = ret.frames, np.reshape(ret.data_approx,data_shape), ret.rel_err_hist
plt.pcolormesh(X,Y,q[:,:,0,5]-qtilde[:,:,0,5])
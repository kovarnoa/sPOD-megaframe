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
sys.path.append('./../lib')
import numpy as np
from numpy import exp, mod,meshgrid
import matplotlib.pyplot as plt
from sPOD_tools import shifted_rPCA, shifted_POD
from transforms import transforms
###############################################################################

##########################################
#%% Define your DATA:
##########################################
plt.close("all")
method = "shifted_POD"
Nx = 400  # number of grid points in x
Nt = 200  # numer of time intervalls

T = 0.5  # total time
L = 1  # total domain size
sigma = 0.015  # standard diviation of the puls
nmodes = 1  # reduction of singular values
x = np.arange(0,Nx)/Nx*L
t =np.arange(0,Nt)/Nt*T
dx = x[1]-x[0]
dt = t[1]-t[0]
c = 1
[X, T] = meshgrid(x, t)
X = X.T
T = T.T
fun = lambda x, t:  exp(-(mod((x-c*t), L)-0.1)**2/sigma**2) + \
                    exp(-(mod((x+c*t), L)-0.9)**2/sigma**2)

# Define your field as a list of fields:
# For example the first element in the list can be the density of
# a flow quantity and the second element could be the velocity in 1D
density = fun(X, T)
velocity = fun(X, T)
shifts1 = np.asarray([-c*t])
shifts2 = np.asarray([c*t])
fields = [density]  #, velocity]

#######################################
# %% CALL THE SPOD algorithm
######################################
data_shape = [Nx,1,1,Nt]
trafos = [transforms(data_shape ,[L], shifts = shifts1, dx = [dx] , use_scipy_transform=True),
            transforms(data_shape ,[L], shifts = shifts2, dx = [dx] , use_scipy_transform=True)]

qmat = np.reshape(fields,[Nx,Nt])
mu = Nx * Nt / (4 * np.sum(np.abs(qmat)))*0.01
if method == "shifted_rPCA":
    ret = shifted_rPCA(qmat, trafos, nmodes_max = np.max(nmodes)+10, eps=1e-16, Niter=300, use_rSVD=True, mu = mu)
else:
    ret = shifted_POD(qmat, trafos, nmodes, eps=1e-16, Niter=300)
sPOD_frames, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
###########################################
# %% 1. visualize your results: sPOD frames
##########################################
# the result is a list of the decomposed field.
# each element of the list contains a frame of the decomposition.<
# If you want to plot the k-th frame use:
# 1. frame
k_frame = 0
plt.figure(num=10)
plt.subplot(121)
plt.pcolormesh(X,T,sPOD_frames[k_frame].build_field())
plt.suptitle("sPOD Frames")
plt.xlabel(r'$N_x$')
plt.ylabel(r'$N_t$')
plt.title(r"$q^"+str(k_frame)+"(x,t)$")
# 2. frame
k_frame = 1
plt.subplot(122)
plt.pcolormesh(X,T,sPOD_frames[k_frame].build_field())
plt.xlabel(r'$N_x$')
plt.ylabel(r'$N_t$')
plt.title(r"$q^"+str(k_frame)+"(x,t)$")
# by default this will plot the field in the first component
# of your field list (here: density)

###########################################
# 2. visualize your results: relative error
##########################################

plt.figure(5)
plt.semilogy(rel_err)
plt.title("relative error")
plt.ylabel(r"$\frac{||X - \tilde{X}_i||_2}{||X||_2}$")
plt.xlabel(r"$i$")

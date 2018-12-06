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
import numpy as np
from numpy import exp, mod, meshgrid
import matplotlib.pyplot as plt
from sPOD_tools import frame, sPOD
###############################################################################

##########################################
#%% Define your DATA:
##########################################
plt.close("all")
Nx = 200  # number of grid points in x
Nt = 250  # numer of time intervalls
dt = 0.01  # size of time intervall
T = Nt*dt  # total time
L = 2*np.pi  # total domain size
x0 = L*0.5  # starting point of the gauss puls
sigma = L/50  # standard deviation of the puls

nmodes = 1  # reduction of singular values
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)
dx = x[1]-x[0]
dt = t[1]-t[0]
c = dx/dt
[T, X] = meshgrid(t, x)

fun = lambda x, t: 0.5 * exp(-(mod((x-c*t), L)-x0)**2/sigma**2) + \
    0.5 * exp(-(mod((x+c*t), L)-x0)**2/sigma**2)

# Define your field as a list of fields:
# For example the first element in the list can be the density of
# a flow quantity and the second element could be the velocity in 1D
density = fun(X, T)
velocity = fun(X, T)
fields = [density]#, velocity]


#######################################
#%% CALL THE SPOD algorithm
######################################
n_velocities = 2  # number of velocities to be detected
sPOD_frames = sPOD(fields, n_velocities, dx, dt, nmodes,
                   eps=1e-4, Niter=30, visualize=False)

#####################################
#%% visualize your results:
#####################################
# the result is a list of the decomposed field.
# each element of the list contains a frame of the decomposition.
# If you want to plot the k-th frame use: 
## 1. frame
k_frame = 0
plt.subplot(121)
sPOD_frames[k_frame].plot_field()
plt.suptitle("sPOD Frames")
plt.xlabel(r'$N_x$')
plt.ylabel(r'$N_t$')
plt.title(r"$q^"+str(k_frame)+"(x,t)$")
## 2. frame
k_frame = 1
plt.subplot(122)
sPOD_frames[k_frame].plot_field()
plt.xlabel(r'$N_x$')
plt.ylabel(r'$N_t$')
plt.title(r"$q^"+str(k_frame)+"(x,t)$")
# by default this will plot the field in the first component 
# of your field list (here: density)

 
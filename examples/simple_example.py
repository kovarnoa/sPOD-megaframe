#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:11:12 2018

@author: philipp
"""

###############################################################################
# IMPORTED MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from sPOD_tools import frame,sPOD
###############################################################################

Nx = 200 # number of grid points in x 
Nt = 250 # numer of time intervalls
dt = 0.01 # size of time intervall
T = Nt*dt # total time
L = 2*np.pi  # total domain size
x0 = L*0.5 # starting point of the gauss puls
sigma=L/50 # standard diviation of the puls
c = 1.3*L/T # speed of the traveling wave

nmodes = 2 # reduction of singular values
x = np.linspace(0,L,Nx)
t = np.linspace(0,T,Nt)
dx=x[1]-x[0]
dt=t[1]-t[0]
c=dx/dt
[X,T] =meshgrid(x,t)
fun = lambda x,t: exp( -(mod((x-c*t),L)-x0)**2/sigma**2 ) + \
                  exp( -(mod((x+c*t),L)-x0)**2/sigma**2 )

field = [fun(X, T)]


plotten=True
eps=1e-4
rel_err=1

sPOD(field, [-c,c], dx, dt, nmodes=2, eps=1e-4, Niter=5, visualize=True)
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


# def load_FOM_data(D,L,Xgrid,Tgrid,T, mu_vecs):
#         from scipy.special import eval_hermite
#
#         # gauss hermite polynomials of order n
#         psi = lambda n, x: (2 ** n * np.math.factorial(n) * np.sqrt(np.pi)) ** (-0.5) * np.exp(-x ** 2 / 2) * eval_hermite(
#             n, x)
#         Nsamples = np.size(mu_vecs,1)
#         w = 0.015 *L
#         Nt = np.size(Tgrid,1)
#         s = np.zeros([Nt,Nsamples])
#         s[:D,:]=mu_vecs
#         shifts = [np.asarray([fft(s[:,n]).imag]) for n in range(Nsamples)]
#         qs1 = []
#         qs2 = []
#         for k in range(Nsamples): # loop over all possible mu vectors
#             q1 = np.zeros_like(Xgrid)
#             q2 = np.zeros_like(Xgrid)
#             for n in range(D): # loop over all components of the vector
#                 q1 += np.exp(-n / 3) * mu_vecs[n, k] * np.sin(2 * np.pi * Tgrid / T * (n+1)) * psi(n, (Xgrid + 0.1 * L) / w)
#                 q2 += np.exp(-n / 3) * mu_vecs[n, k] * np.sin(2 * np.pi * Tgrid / T * (n+1)) * psi(n, (Xgrid - 0.1 * L) / w)
#
#             qs1.append( q1)
#             qs2.append(-q2)
#
#
#         q1 = np.concatenate(qs1, axis=1)
#         q2 = np.concatenate(qs2, axis=1)
#         q_frames = [q1, q2]
#
#         shifts = [np.concatenate(shifts, axis=1), -np.concatenate(shifts, axis=1)]
#         data_shape = [Nx, 1, 1, Nt * Nsamples]
#         trafos = [transforms(data_shape, [L], shifts=shifts[0], dx=[dx], use_scipy_transform=True),
#                   transforms(data_shape, [L], shifts=shifts[1], dx=[dx], use_scipy_transform=True)]
#
#         q = 0
#         for trafo, qf in zip(trafos, q_frames):
#             q += trafo.apply(qf)
#
#         return q, q1, q2, shifts, trafos



plt.close("all")
data = loadmat('a30b10/vort.mat')
data = data['data'][:,:,::2,::2]

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
rotation1   = pi/6 * np.cos(2*pi*time) + pi/18 * np.sin(2*pi*time)    # frame 1
shift1      = np.zeros([2,Nt])
shift2      = np.zeros([2,Nt])
rotation2   = np.zeros([Nt])
rotation2   = pi/6 * np.cos(2*pi*time) + pi/18 * np.sin(2*pi*time)   # frame 2

q = np.zeros(data_shape)
#for nvar in range(Nvar):
for it,t in enumerate(time):
    q[:,:,0,it] = np.array(data[0,it,:,:]).T
    plt.pause(0.01)
    shift1[0,it] = 0.3*L[0]
    shift1[1,it] = -0.025 * np.cos(2*pi*t)
    shift2[0,it] = 0.2*L[0]
    shift2[1,it] = 0


# %% Create Trafo
trafo_1 = transforms(data_shape,L, trafo_type="shiftRot", shifts = shift1, dx = [dx,dy], rotations = rotation1,  use_scipy_transform=True)
trafo_2 = transforms(data_shape,L, trafo_type="shiftRot", shifts = shift2, dx = [dx,dy], rotations = rotation2,  use_scipy_transform=True)


qshift1 = trafo_1.reverse(q)
qshift2 = trafo_2.reverse(q)
qmin = np.min(np.reshape(qshift2,[-1]))
qmax = np.max(np.reshape(qshift2,[-1]))
clim = max(abs(qmin),abs(qmax))*0.5
for it in range(0,len(time),2):
    p=1  # resolution
    plt.subplot(1,2,1)
    plt.pcolormesh(X[::p,::p],Y[::p,::p],qshift1[::p,::p,0,it],cmap=cm,vmin=-clim,vmax=clim)
    plt.title("frame 1")
    plt.subplot(1,2,2)
    plt.pcolormesh(X[::p,::p],Y[::p,::p],qshift2[::p,::p,0,it],cmap=cm,vmin=-clim,vmax=clim)
    plt.title("frame 2")
    plt.colorbar()
    plt.pause(0.01)
qshiftreverse = trafo_1.apply(trafo_1.reverse(q))
res = q-qshiftreverse
#
# for it in range(0,len(time),5):
#     plt.pcolormesh(X,Y,res[...,0,it])
#     plt.pause(0.001)
err = np.linalg.norm(np.reshape(res,-1))/np.linalg.norm(np.reshape(q,-1))
err_hist = [np.linalg.norm(np.reshape(res[...,it],-1))/np.linalg.norm(np.reshape(q[...,it],-1)) for it in range(len(time))]
print("err =  %4.4e "% err)
#plt.pcolormesh(X,Y,q[...,0,0]-qshiftreverse[...,0,0],cmap = cm)
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
r=2*nmodes
[U, S, VT] = np.linalg.svd(qmat, full_matrices=False)
qmat_tilde =np.dot(U[:, :r] * S[:r], VT[:r,:])
print("SVD rel err:", np.linalg.norm(qmat -qmat_tilde)/np.linalg.norm(qmat))
ret = shifted_POD(qmat, transforms, nmodes=nmodes, eps=1e-4, Niter=40)

mu = np.prod(np.size(qmat)) / (4 * np.sum(np.abs(qmat)))
#ret = shifted_rPCA(qmat, transforms, eps=1e-4, Niter=50, visualize=False, mu = mu)
qframes, qtilde = ret.frames, ret.data_approx
qtilde = np.reshape(qtilde,data_shape)
plt.pcolormesh(X,Y,q[...,0,10]-qtilde[...,0,10])


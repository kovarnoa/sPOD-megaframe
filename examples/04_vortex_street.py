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
from lib.plot_utils import show_animation, save_fig
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 24}

matplotlib.rc('font', **font)
###############################################################################
cm = farge_colormap_multi( etalement_du_zero=0.2, limite_faible_fort=0.5)
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
nmodes = [40,40]                              # reduction of singular values
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
#show_animation(np.squeeze(qshift2),Xgrid=[X,Y])
qshiftreverse = shift_trafo_2.apply(qshift2)
res = q-qshiftreverse
err = np.linalg.norm(np.reshape(res,-1))/np.linalg.norm(np.reshape(q,-1))
print("err =  %4.4e "% err)
plt.pcolormesh(X,Y,q[...,0,0]-qshiftreverse[...,0,0])
plt.colorbar()

# %% Run shifted POD
trafos = [shift_trafo_1, shift_trafo_2]
qmat = np.reshape(q, [-1, Nt])


#ret = shifted_POD(qmat, trafos, nmodes=nmodes, eps=1e-4, Niter=100, use_rSVD=True)
[N,M]= np.shape(qmat)
mu0 = N * M / (4 * np.sum(np.abs(qmat)))*0.0001
#lambd0 = 1 / np.sqrt(np.maximum(M, N))
lambd0 = mu0*1e2
ret = shifted_rPCA(qmat, trafos, nmodes_max = np.max(nmodes)+200, eps=1e-10, Niter=100, visualize=True, use_rSVD=True, lambd=lambd0, mu=mu0)
# NmodesMax = 30
# rel_err_matrix = np.ones([30]*len(trafos))
# for r1 in range(0,NmodesMax):
#     for r2 in range(0,NmodesMax):
#         print("mode combi: [", r1,r2,"]\n")
#         ret = shifted_POD(qmat, trafos, nmodes=[r1,r2], eps=1e-4, Niter=40, use_rSVD=True)
#         print("\n\nmodes: [", r1, r2, "] error = %4.4e \n\n"%ret.rel_err_hist[-1])
#         rel_err_matrix[r1,r2] = ret.rel_err_hist[-1]


qframes, qtilde , rel_err_list = ret.frames, np.reshape(ret.data_approx,data_shape), ret.rel_err_hist
qf = [np.reshape(trafo.apply(frame.build_field()),data_shape) for trafo,frame in zip(trafos,qframes)]
E = np.reshape(ret.error_matrix,data_shape)


lims = [-1,1]
nt = 250


fig, ax = plt.subplots(1,4,num=10)
h_list = [0]*4
h_list[0] = ax[0].pcolormesh(X, Y, np.squeeze(qtilde[..., nt]), cmap=cm)
ax[0].set_title(r"$\tilde q$")
h_list[1] = ax[1].pcolormesh(X, Y, np.squeeze(qf[0][..., nt]), cmap=cm)
ax[1].set_title(r"$T^{\Delta_1} q_1$")
h_list[2] = ax[2].pcolormesh(X, Y, np.squeeze(qf[1][..., nt]), cmap=cm)
ax[2].set_title(r"$T^{\Delta_2} q_2$")
h_list[3] = ax[3].pcolormesh(X, Y, np.squeeze(E[..., nt]), cmap=cm)
ax[3].set_title(r"$e$")
for a,h in zip(ax,h_list):
    h.set_clim(lims)
    a.axis("image")
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel(r"$x$")
    a.set_ylabel(r"$y$")

cax = fig.add_axes([a.get_position().x1+0.01,a.get_position().y0,0.02,a.get_position().height])
plt.colorbar(h, cax=cax)


#np.savetxt('sPOD_error.txt', rel_err_matrix)

#

# plt.pcolormesh(X,Y,q[:,:,0,5]-qtilde[:,:,0,5])
###########################
# error of svd:
[U, S, VT] = np.linalg.svd(qmat, full_matrices=False)
err_svd = np.sqrt(1-np.cumsum(S[:60]**2)/np.sum(S**2))
###########################
rank = 160
u = U[:, :rank]
s = S[:rank]
vh = VT[:rank, :]
# add up all the modes A=U * S * VH
qsvd = np.reshape(np.dot(u * s, vh),data_shape)
fig, ax = plt.subplots(1,2,num=11)
h=[0,1]
h[0] = ax[0].pcolormesh(X, Y, np.squeeze(q[..., nt]), cmap=cm)
ax[0].set_title(r"$q(x,t_i)$")
h[1] = ax[1].pcolormesh(X, Y, np.squeeze(qsvd[..., nt]), cmap=cm)
ax[1].set_title(r"$\tilde q(x,t_i)$")
for a,h in zip(ax,h):
    h.set_clim(lims)
    a.axis("image")
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel(r"$x$")
    a.set_ylabel(r"$y$")

cax = fig.add_axes([a.get_position().x1+0.01,a.get_position().y0,0.02,a.get_position().height])
plt.colorbar(h, cax=cax)
#
with open('sPOD_error.txt', 'r') as f:
    rel_err_matrix = [[float(num) for num in line.split(' ')] for line in f]

rel_err_matrix = np.asarray(rel_err_matrix)
smallest_error = np.ones([30+30,1])
index = ['a']*60
for i in range(30):
    for k in range(30):
        if smallest_error[i+k]>rel_err_matrix[i,k]:
            smallest_error[i+k] = rel_err_matrix[i,k]
            index[i+k] = '%d+%d'%(i,k)

plt.figure(3)
plt.plot(smallest_error[:-1],'*', label="sPOD")
plt.plot(err_svd,'x-',label="POD")
plt.xticks(np.arange(1,60,4),index[1::4])
plt.legend()
plt.xlabel(r"shifted ranks $r_1+r_2$ ")
plt.ylabel(r"error")



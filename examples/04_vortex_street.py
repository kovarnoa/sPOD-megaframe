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
from utils import *
from numpy.fft import fft2,ifft2
from farge_colormaps import farge_colormap_multi
from lib.plot_utils import show_animation, save_fig
from utils import *
import matplotlib
from os.path import expanduser
home = expanduser("~")

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 24}

matplotlib.rc('font', **font)
###############################################################################
cm = farge_colormap_multi( etalement_du_zero=0.2, limite_faible_fort=0.5)
#from sympy.physics.vector import curl
###############################################################################

##########################################
#% Define your DATA:
##########################################
plt.close("all")
dir = home+"/develop/data/two_cylinders/20211018_one_cylinder_moving/"
dir = home+"/develop/data/two_cylinders/20210927_two_cylinders_a0.25/"
path = home+"/develop/data/two_cylinders/20211020_pathopt.mat"
#dir = home+"/develop/data/two_cylinders/20211019_big_domain/"

data = loadmat(path)
fields = data["data"]
mask = fields[0,...].T
p = fields[1,...].T
ux = fields[2,...].T
uy = fields[3,...].T
time = data["time"].flatten()
time = time - time[0]
# %%

Nt = len(time)                      # Number of time intervalls
Nvar = 1# data.shape[0]                    # Number of variables
nmodes = [40,40]                              # reduction of singular values
frac = 4


Ngrid = [fields.shape[2]//frac, fields.shape[3]//frac]  # number of grid points in x
data_shape = [*Ngrid,Nvar,2*Nt]
               # size of time intervall
freq    = 0.01/5
L = data["domain_size"][0]   # total domain size
T = time[-1]
x,y = (np.linspace(0, L[i], Ngrid[i]) for i in range(2))
dX = (x[1]-x[0],y[1]-y[0])
dt = time[1]-time[0]
[Y,X] = meshgrid(y,x)
fd = finite_diffs(Ngrid,dX)

vort= np.asarray([fd.rot(ux[::frac,::frac, nt],uy[::frac,::frac,nt]) for nt in range(np.size(ux,2))])
vort = np.moveaxis(vort,0,-1)
q = np.zeros(data_shape)
#for nvar in range(Nvar):
#for it,t in enumerate(time):
#    q[:,:,0,it] = np.array(data[0,it,::frac,::frac]).T

data = q
q = np.concatenate([ux[::frac,::frac,:],uy[::frac,::frac,:]],axis=-1)
time = np.concatenate([np.linspace(0, T, Nt),np.linspace(0, T, Nt)],axis=0)
# %%data = np.zeros(data_shape)
#for tau in range(0,Nt):
#    data[0,tau,:,:] = curl(np.squeeze(ux[0,tau,:,:]),np.squeeze(uy[0,tau,:,:]))

               # size of time intervall

shift1 = np.zeros([2,2*Nt])
shift2 = np.zeros([2,2*Nt])
shift1[0,:] = 0 * time                      # frame 1, shift in x
shift1[1,:] = 0 * time                      # frame 1, shift in y
shift2[0,:] = 0 * time                      # frame 2, shift in x
shift2[1,:] = -(8.2*np.sin(2*pi*freq*time)+8.2*np.sin(4*pi*freq*time)+8.2*np.sin(6*pi*freq*time)) # frame 2, shift in y

# %% Create Trafo

shift_trafo_1 = transforms(data_shape,L, shifts = shift1,trafo_type="identity", dx = dX, use_scipy_transform=False )
shift_trafo_2 = transforms(data_shape,L, shifts = shift2, dx = dX, use_scipy_transform=True )
qshift1 = shift_trafo_1.reverse(q)
qshift2 = shift_trafo_2.reverse(q)
#show_animation(np.squeeze(qshift2),Xgrid=[X,Y])
qshiftreverse = shift_trafo_2.apply(qshift2)
res = q-qshiftreverse
err = np.linalg.norm(np.reshape(res,-1))/np.linalg.norm(np.reshape(q,-1))
err_time = [np.linalg.norm(np.reshape(res[...,i],-1))/np.linalg.norm(np.reshape(q[...,i],-1)) for i in range(len(time))]
print("err =  %4.4e "% err)
plt.pcolormesh(X,Y,q[...,34]-qshiftreverse[...,34])
plt.colorbar()

# %% Run shifted POD
trafos = [shift_trafo_1, shift_trafo_2]
qmat = np.reshape(q, [-1, 2*Nt])


#ret = shifted_POD(qmat, trafos, nmodes=nmodes, eps=1e-4, Niter=100, use_rSVD=True)
[N,M]= np.shape(qmat)
mu0 = N * M / (4 * np.sum(np.abs(qmat)))*0.001
lambd0 = 1 / np.sqrt(np.maximum(M, N))*10
#lambd0 = mu0*5e2
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



# %%
lims = [-1,1]
nt = 200
vort_tilde = fd.rot(qtilde[..., 0, nt],qtilde[..., 0, nt+Nt])
vort_f1 = fd.rot(qf[0][..., 0, nt],qf[0][..., 0, nt+Nt])
vort_f2 = fd.rot(qf[1][..., 0, nt],qf[1][..., 0, nt+Nt])
vort_E = fd.rot(E[..., 0, nt],E[..., 0, nt+Nt])
qplot = [vort_tilde,vort_f1,vort_f2, vort_E]
fig, ax = plt.subplots(1,3,num=10)
h_list = [0]*3
cm = farge_colormap_multi( etalement_du_zero=0.2, limite_faible_fort=0.5)
h_list[0] = ax[0].imshow( qplot[0].T, cmap=cm)
ax[0].set_title(r"sPOD")
h_list[1] = ax[1].imshow( qplot[1].T, cmap=cm)
ax[1].set_title(r"Frame 1")
h_list[2] = ax[2].imshow( qplot[2].T, cmap=cm)
ax[2].set_title(r"Frame 2")
#h_list[3] = ax[3].imshow( qplot[3].T, cmap=cm)
#ax[3].set_title(r"Noise")
for a,h in zip(ax,h_list):
    h.set_clim(lims)
    a.axis("image")
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel(r"$x$")
    a.set_ylabel(r"$y$")

cax = fig.add_axes([a.get_position().x1+0.01,a.get_position().y0,0.02,a.get_position().height])
plt.colorbar(h, cax=cax)
save_fig("imgs/vortex_rPCA.png")

#np.savetxt('sPOD_error.txt', rel_err_matrix)

# %%
rank = np.sum(ret.ranks)
# plt.pcolormesh(X,Y,q[:,:,0,5]-qtilde[:,:,0,5])
###########################
# error of svd:
qmat2 = np.reshape(qshift2,[-1,Nt])
[U, S, VT] = np.linalg.svd(qmat, full_matrices=False)
err_svd = np.sqrt(1-np.cumsum(S[:rank+5]**2)/np.sum(S**2))
###########################

print("error svd: %2.2e"%err_svd[rank])
u = U[:, :rank]
s = S[:rank]
vh = VT[:rank, :]
# add up all the modes A=U * S * VH
qsvd = np.reshape(np.dot(u * s, vh),data_shape)
vort_svd=fd.rot(qsvd[..., 0, nt],qsvd[..., 0, nt+Nt])
fig, ax = plt.subplots(1,2,num=11)
h=[0,1]
h[0] = ax[0].imshow( vort[...,nt].T, cmap=cm)
ax[0].set_title(r"data")
h[1] = ax[1].imshow(vort_svd.T, cmap=cm)
ax[1].set_title(r"POD")
for a,h in zip(ax,h):
    h.set_clim(lims)
    a.axis("image")
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel(r"$x$")
    a.set_ylabel(r"$y$")

cax = fig.add_axes([a.get_position().x1+0.01,a.get_position().y0,0.02,a.get_position().height])
plt.colorbar(h, cax=cax)


save_fig("imgs/vortex_POD.png")
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



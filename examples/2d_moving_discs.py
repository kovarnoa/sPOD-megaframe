#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D MOVING DISCS

Created on Sat Jan  2 14:52:26 2021

@author: phil
"""

###############################################################################
# IMPORTED MODULES
###############################################################################
import sys
sys.path.append('../lib')
import numpy as np
from numpy import exp, mod,meshgrid,pi,sin,size
import matplotlib.pyplot as plt
from sPOD_tools import frame, sPOD_distribute_residual
from scipy import sparse
###############################################################################

##########################################
#%% Define your DATA:
##########################################
plt.close("all")
Ngrid = [401, 202]  # number of grid points in x
Nt = 50            # Number of time intervalls
Nvar = 1            # Number of variables
nmodes = 1          # reduction of singular values

data_shape = [*Ngrid,Nvar,Nt]
               # size of time intervall
T = 2*pi                # total time
L = np.asarray([2, 1])   # total domain size
x0 = L*0.5               # starting point of the gauss puls
R = 0.1 * min(L)         # Radius of cylinder
x,y = (np.linspace(0, L[i], Ngrid[i]) for i in range(2))
time = np.linspace(0, T, Nt)
dx,dy = (x[1]-x[0],y[1]-y[0])
dt = time[1]-time[0]
c = dx/dt
[Y,X] = meshgrid(y,x)


f = lambda x,l : ((np.tanh(x/l) + 1 ) * 0.5)

phi1 = np.zeros(data_shape[:-2])
phi2 = np.zeros(data_shape[:-2])
q = np.zeros(data_shape)

shift1 = np.zeros([2,Nt])
shift2 = np.zeros([2,Nt])

center1 = (0.2*L[0],0.5*L[1])
center2 = (0.5*L[0],0.5*L[1])

for it,t in enumerate(time):
    
    x1,y1 = (center1[0], 0.3*L[1]*sin(t) + center1[1])
    x2,y2 = (center2[0]-0.3*L[1]*sin(t), - 0.3*L[1]*sin(t) + center2[1])
    
    phi1 = np.sqrt((X-x1)**2 + (Y-y1)**2) - R
    phi2 = np.sqrt((X-x2)**2 + (Y-y2)**2) - R
    
    shift1[0,it] = x1-center1[0]
    shift1[1,it] = y1-center1[1]
    
    shift2[0,it] = x2-center2[0]
    shift2[1,it] = y2-center2[1]
    
    q[...,0,it] = f(phi1,dx*3)*f(phi2,dx*3) 
    #q[...,1,it] = f(phi1,dx)-f(phi2,dx) 
    #plt.pcolormesh(X,Y,q[...,1,it])
    #plt.show()
    #plt.pause(0.001)
    

# %%
#########################
# Lagrange interpolation    
#########################
def lagrange(xvals, xgrid, j):
    """
    Returns the j-th basis polynomial evaluated at xvals
    using the grid points listed in xgrid
    """    
    xgrid = np.asarray(xgrid)
    if not isinstance(xvals,list):
        xvals=[xvals]
    n = np.size(xvals)
    Lj = np.zeros(n)
    for i,xval in enumerate([xvals]):
        nominator = xval - xgrid
        denominator = xgrid[j] - xgrid
        p = nominator/denominator
        p[j] = 1
        Lj[i] = np.prod(p)

    return Lj
    
    
    
# %%    
###############################################################################
# CLASS of Transformation applied to the field
###############################################################################                                
class Transform:
    # TODO: Add properties of class frame in the description
    """ Class of all Transforms.
        A transformation can be implemented as a 
            + shift T^c q(x,t) = q(x-ct,t)
            + rotation T^w q(x,y,t) = q(M(omega)(x,y),t) where M(omega) is the rotation matrix
    """

    def __init__(self, data_shape, domain_size, trafo_type="shift", shifts = None, dx = None ):
        self.Ngrid = data_shape[:2]
        self.Nvar = data_shape[2]
        self.Ntime = data_shape[3]
        self.data_shape = data_shape
        self.domain_size = domain_size
        if trafo_type=="shift":
            self.shifts = shifts    # dim x Ntime shiftarray (one element for one time instance)
            self.dx = dx            # list of lattice spacings
            self.dim = size(dx)
            self.shiftMatrices_pos, self.shiftMatrices_neg = self.init_shifts() # list of shiftoperators (one element for one time instance)

        
    
    def apply(self, field):
        """
        This function returns the shifted field.
        $q(x-s,t)=T^s[q(x,t)]$ 
        here the shift is simply s=c*t
        In the default case where c= ~velocity of the frame~,
        the field is shifted back to the original frame. 
        ( You may call it the labratory frame)
        Before we shift the frame has to be put togehter in the co-moving
        frame. This can be done by build_field().
        
        """
        return self.shift(field, self.shiftMatrices_pos)
        

    def reverse(self, field):
        """
        This function returns the shifted field.
        $q(x-s,t)=T^s[q(x,t)]$ 
        here the shift is simply s=c*t
        In the default case where c= ~velocity of the frame~,
        the field is shifted back to the original frame. 
        ( You may call it the labratory frame)
        Before we shift the frame has to be put togehter in the co-moving
        frame. This can be done by build_field().
        """
        return self.shift(field, self.shiftMatrices_neg)

    def shift(self, field, shiftMatrices):
        """
        This function returns the shifted field.
        $q(x-s,t)=T^s[q(x,t)]$ 
        here the shift is simply s=c*t
        In the default case where c= ~velocity of the frame~,
        the field is shifted back to the original frame. 
        ( You may call it the labratory frame)
        Before we shift the frame has to be put togehter in the co-moving
        frame. This can be done by build_field().
        
        """
        Ntime = np.size(field,-1)
        field_shift = np.zeros_like(field)
        for it in range(Ntime):
            vec = np.reshape(field[...,it],-1)
            vec_shift = shiftMatrices[it]@vec
            field_shift[...,it] = np.reshape(vec_shift,self.data_shape[:-1])

        return field_shift
    
    def init_shifts(self):
        ### implement pos shift matrix ###
        shift_pos_mat_list = []
                
        Nx, Ny = self.Ngrid
        Lx, Ly = self.domain_size
        dx, dy = self.dx
        
        shiftx_pos = self.shifts[0,:]
        shifty_pos = self.shifts[1,:]
        
        shiftx_pos_mat_list=self.compute_shift_matrix(shiftx_pos, Lx, dx, Nx)
        shifty_pos_mat_list=self.compute_shift_matrix(shifty_pos, Ly, dy, Ny)
        
        # kron for each time slice
        for shiftx,shifty in zip(shiftx_pos_mat_list,shifty_pos_mat_list):
            shift_pos_mat_list.append(sparse.kron(shiftx, shifty))
        
        ### implement neg shift matrix ###
        shift_neg_mat_list = []
        
        shiftx_neg = -shiftx_pos +dx
        shifty_neg = -shifty_pos +dy
        shiftx_neg_mat_list=self.compute_shift_matrix(shiftx_neg, Lx, dx, Nx)
        shifty_neg_mat_list=self.compute_shift_matrix(shifty_neg, Ly, dy, Ny)
        
        # kron for each time slice
        for shiftx,shifty in zip(shiftx_neg_mat_list,shifty_neg_mat_list):
            shift_neg_mat_list.append(sparse.kron(shiftx, shifty))
        
        return shift_pos_mat_list, shift_neg_mat_list
        
    def compute_shift_matrix(self,shift_list, domain_length, spacing, Npoints):
        from numpy import floor
        
        Mat = []
        for shift in shift_list:
            
            # we assume periodicity here
            if shift > domain_length:
                shift = shift - domain_length
            elif shift < 0:
                shift = shift + domain_length
        
            ''' interpolation scheme        lagrange_idx(x)= (x-x_{idx-1})/(x_idx - x_0)+
            -1      0   x    1       2                    ...+(x-x_{idx+2})/(x_idx - x_{idx+2})
             +      +   x    +       +
           idx_m1  idx_0    idx_1   idx_2
          =idx_0-1        =idx_0+1
           '''
           
            # shift is close to some discrete index:
            idx_0 = floor(shift/spacing)
            # save all neighbours
            idx_list = np.asarray([idx_0-1, idx_0, idx_0+1, idx_0+2],dtype=np.int32)
            
            if idx_list[0] < 0 : idx_list[0] += Npoints 
            if idx_list[3] > Npoints-1 : idx_list[3] -= Npoints
            # subdiagonals needed if point is on other side of domain
            idx_subdiags_list = idx_list - Npoints
            # compute the distance to the index
            delta_idx = shift/spacing - idx_0
            # compute the 4 langrage basis elements
            lagrange_coefs = [lagrange(delta_idx, [-1,0,1,2], j) for j in range(4)]
            # for the subdiagonals as well
            lagrange_coefs = lagrange_coefs +lagrange_coefs
            
            # band diagonals for the shift matrix
            offsets = np.concatenate([idx_list,idx_subdiags_list])
            diagonals = [np.ones(Npoints)*Lj  for Lj in lagrange_coefs]

            Mat.append(sparse.diags(diagonals,offsets,shape=[Npoints,Npoints]))
        
        return Mat
# %% Create Trafo

shift_trafo_1 = Transform(data_shape,L, shifts = shift1, dx = [dx,dy] )
shift_trafo_2 = Transform(data_shape,L, shifts = shift2, dx = [dx,dy] )
qshift1 = shift_trafo_1.apply(q)
qshift2 = shift_trafo_2.apply(q)
qshiftreverse = shift_trafo_2.reverse(shift_trafo_2.apply(q))
res = q-qshiftreverse
err = np.linalg.norm(np.reshape(res,-1))/np.linalg.norm(np.reshape(q,-1))
print("err =  %4.4e "% err)
plt.pcolormesh(X,Y,q[...,0,0]-qshiftreverse[...,0,0])
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
transforms = [shift_trafo_1, shift_trafo_2]
qframes, q = sPOD_distribute_residual(q, transforms, nmodes=2, eps=1e-4, Niter=20, visualize=True)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:19:19 2018

@author: Philipp Krah, Jiahan Wang

This package provides all the infrastructure for the 
    
    shifted propper orthogonal decomposition (SPOD)

"""
############################
# import MODULES here:
############################
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, meshgrid, mod,size, interp, where, diag, reshape, \
                    asarray
from numpy.linalg import svd, lstsq, norm
from matplotlib.pyplot import   subplot, plot, pcolor, semilogy, title, \
                                xlabel, ylabel, figure \

###############################
# sPOD general SETTINGS:
###############################
# latex font for plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
##############################

# %%
###############################################################################
# CLASS of CO MOVING FRAMES
###############################################################################                                
class frame:
    """ Definition is physics motivated: 
        All points are in a inertial system (frame), if they can be transformed to 
        the rest frame by a galilei transform.
        The frame is represented by an orthogonal system.
    """

    def __init__(self, velocity, dx, dt, fields, number_of_modes=1):
        """
        Initialize a co moving frame.
        """
        # TODO in future versions the velocity
        # will be replaced by the transformation mapping
        self.velocity = velocity # its actually the relative velocity to the labframe
        self.Nmodes = number_of_modes
        self.set_orhonormal_system(dx,dt,fields)
    #    print("We have initialiced a new field!")
    
    def shift(self, field):
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
        c=self.velocity
        idx_shift = c*self.dt/self.dx
        Ix = range(size(field, 1))
        qshift = field.copy()
        for j in range(size(field,0)):
             #roll(qshift[j,:],int(idx_shift)*j)
            qshift[j,:] = interp(Ix-idx_shift*j, Ix, qshift[j,:], period=Ix[-1])
        
        return qshift


    def reduce(self, field):
        """
        Reduce the full filed using the first N modes of the Singular
        Value Decomposition
        """
        Nmodes = self.Nmodes
        [U, S, V] = svd(field, full_matrices=True)
        Sr = S[:Nmodes]
        Ur = U[:, :Nmodes]
        Vr = V[:Nmodes, :]
        
        return Ur, Sr, Vr
        
    def set_orhonormal_system(self, dx, dt, fields):
        """
         
        """
        
        # spatial lattice spacing
        self.dx = dx
        # timestep
        self.dt = dt
        # field shape
        self.field_shape = np.shape(fields)
        # number of data fields
        self.Nfields = size(fields, 0)
        # number of space dimensions
        self.dim = fields[0].ndim - 1
        # list of spatial points in each dimension
        self.Nspace = np.ones(2)
        self.Nspace[:self.dim] = fields[0].shape[:self.dim]
        # number of timesteps
        self.Ntime = fields[0].shape[-1]

        # init field
        X = []

        # loop
        for idx_field,field in enumerate(fields):
            field_shift = self.shift(field)
            field_shift = reshape(field_shift, [-1, self.Ntime])
            X.append(field_shift)
            
        [U, S, VT] = self.reduce(X[0])
        self.modal_system = {"U" : U, "sigma" : S, "VT": VT}
  
    def build_field(self):
        # modes from the singular value decomposition
        u = self.modal_system["U"]
        s = self.modal_system["sigma"]
        vh= self.modal_system["VT"]
        # add up all the modes A=U * S * VH
        return np.reshape(np.dot(u * s, vh),self.field_shape)
    
    def plot_field(self,field_index=0):
        
        pcolor(self.shift(self.build_field()[field_index]))

    def plot_singular_values(self):
        """
        This function plots the singular values of the frame.
        """
        sigmas=self.modal_system["sigma"]        
        semilogy(sigmas,"r+")
        xlabel("i")
        ylabel("$\sigma_i$")
        
    def __add__(self, other):
        """ Add two frames for the purpose of concatenating there modes """
        # TODO make check if other and self can be added:
        # are they in the same frame? Are they from the same data etc.
        new=frame(self.velocity,self.dx,self.dt,self.build_field(),self.Nmodes)
        # combine left singular vecotrs
        Uself = self.modal_system["U"]
        Uother = other.modal_system["U"]
        new.modal_system["U"]=np.concatenate([Uself,Uother],axis=1)

        # combine right singular vecotrs
        VTself = self.modal_system["VT"]
        VTother = other.modal_system["VT"]
        new.modal_system["VT"]=np.concatenate([VTself,VTother],axis=0)

        Sself=self.modal_system["sigma"]
        Sother=other.modal_system["sigma"]
        new.modal_system["sigma"]=np.concatenate([Sself,Sother])

        new.Nmodes += other.Nmodes 
        
        return new


# %%
###############################################################################
# least square minimization
###############################################################################

def minimize(Xtilde_frames,X):
  ######################################################
    # build coef matrix X_coef_mat of eq. (9) in Reiss2017
    ######################################################
    X_coef=[]       # lab frame
    X_coef_shift=[] # co-moving frame
    dx=Xtilde_frames[0].dx
    dt=Xtilde_frames[0].dt
    # number of moving frames
    Nframes = len(Xtilde_frames)
    # loop through all moving frames
    for k,frame in enumerate(Xtilde_frames):
        # left singular vectors
        U=frame.modal_system["U"]
        # right singular vectors 
        VT=frame.modal_system["VT"]
        # singular values
        sigmas=frame.modal_system["sigma"]
        Nmodes=frame.Nmodes
        # velocity in the corresponding frame
        c = frame.velocity
        
        # loop over all the modes
        # each mode represents one alpha_k and one column of the matrix X
        for i in range(Nmodes):
            Xmode_shift = np.outer(U[:,i],VT[i,:])    
            Xmode = frame.shift(Xmode_shift)
            
            X_coef_shift.append(reshape(Xmode_shift,[-1]))
            X_coef.append(reshape(Xmode,[-1]))
    ######################################################
    # solve the minimication problem 
    ######################################################
    X_coef = asarray(X_coef).T
    X_coef_shift = asarray(X_coef_shift)       
    X_ref = reshape(X.copy(), [-1, 1])
    # X_coef * alpha = X
    alpha = lstsq(X_coef, X_ref)[0]
    alpha = asarray(alpha)
    

    for k,frame in enumerate(Xtilde_frames):
        X_frame = X_coef_shift.T[:,k*Nmodes:(k+1)*Nmodes] @ alpha[k*Nmodes:(k+1)*Nmodes] 
        X_frame = reshape(X_frame, np.shape(X[0]))
        [U, S, VT]=frame.reduce( X_frame )
        frame.modal_system = {"U" : U, "sigma" : S, "VT": VT}


# %%
###############################################################################
# sPOD algorithm
###############################################################################


def sPOD(X, velocities, dx, dt, nmodes=2, eps=1e-4, Niter=5, visualize=True):
    
    # plot the first component of the original field
    if visualize:
        subplot(1, 4, 1)
        p = pcolor(X[0])
        plt.title(r'$X$')
        plt.ylabel(r"$N_t$")
        plt.xlabel(r"$N_x$")
        plt.pause(0.05)
        clims = p.get_clim()
        
    #################################
    # 1. reset loop variables
    ################################
    
    Xtilde=np.zeros(np.shape(X))
    Xtilde_frames=[frame(v,dx,dt,Xtilde,nmodes) for v in velocities]
    rel_err=1
    it=0
    
    ###########################################################################
    # MAIN LOOP
    ###########################################################################
    # loop until the desired precission is achieved or the maximal number
    # of iterations
    while rel_err > eps and it < Niter :
        
        it+=1 # counts the number of iterations in the loop

        
        ###############################
        # 2. calculate residual R
        ###############################
        R = X - Xtilde
        rel_err = norm(R)/norm(X) # relative error

        ##########################################
        print("iter= ", it, "rel err= ", rel_err)
        ##########################################
        
        if visualize: # plot the residual
            subplot(1,len(Xtilde_frames)+2,2)
            pcolor(R[0])   
            plt.title(r"$R=X-\tilde{X}$")    
            #plt.xlabel(r"$N_x$")
            plt.pause(0.05)

        ###############################
        # 3. Multi shift and reduce R
        ###############################
        R_frames=[frame(v, dx, dt, R, nmodes) for v in velocities]
    
        #######################################################################
        # 4. combine the modes of Xtilde and R
        # Note 2 objects in the same frame, can be added by concatenating 
        # the SVD left and right singular vectors.
        #######################################################################
        #Xtilde_frames = []
        for k, R_frame in enumerate(R_frames):
            Xtilde_frames[k]=R_frame + Xtilde_frames[k]
        ###################################################
        # 5. Solve (Xtilde+R) * alpha = X for unknown alpha
        #  + the classical method (Reiss2018) uses the 
        #    least square approach
        # TODO Implement the other methods as well
        ###################################################
        minimize(Xtilde_frames, X)
        
        #############################################
        # 5. Add up all the frames to compute Xtilde
        #############################################
        Xtilde *= 0
        for k, Xframe in enumerate(Xtilde_frames):

            Xtilde+=Xframe.shift(Xframe.build_field()[0])
            
            # we plot the first 3 frames 
            if visualize and k <= 3:
                subplot(1,len(Xtilde_frames)+2,k+3)
                Xframe.plot_field()
                plt.clim(clims)
                plt.title(r"$q^"+str(k)+"(x,t)$")        
                #plt.xlabel(r"$N_x$")
            plt.pause(0.05)

    ###########################################################################
    # End of MAIN LOOP
    ###########################################################################
   
    
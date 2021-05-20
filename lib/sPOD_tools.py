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
import time
from matplotlib.pyplot import   subplot, plot, pcolor, semilogy, title, \
                                xlabel, ylabel, figure \

###############################
# sPOD general SETTINGS:
###############################

# %%
###############################################################################
# CLASS of CO MOVING FRAMES
###############################################################################                                
class frame:
    # TODO: Add properties of class frame in the description
    """ Definition is physics motivated: 
        All points are in a inertial system (frame), if they can be transformed to 
        the rest frame by a galilei transform.
        The frame is represented by an orthogonal system.
    """

    def __init__(self, transform, field=None, number_of_modes=1):
        """
        Initialize a co moving frame.
        """
        data_shape = np.shape(field)
        self.Ngrid = data_shape[:2]
        self.Nvar = data_shape[2]    # number of state variables [rho,u,v,p] 
        self.Ntime = data_shape[3]
        self.data_shape = data_shape
        self.trafo = transform
        self.dim = self.trafo.dim
        self.Nmodes = number_of_modes
        # transform the field to reference frame
        if not np.all(field == None):
            field = self.trafo.apply(field)
            self.set_orhonormal_system(field)
        #print("We have initialiced a new field!")
    


    def reduce(self, field, r):
        """
        Reduce the full filed using the first N modes of the Singular
        Value Decomposition
        """
        [U, S, V] = svd(field, full_matrices=False)
        Sr = S[:r]
        Ur = U[:, :r]
        Vr = V[:r, :]
        
        return Ur, Sr, Vr
        
    def set_orhonormal_system(self, field):
        """
        In this routine we set the orthonormal vectors of the SVD in the 
        corresponding frames.
        """

        # reshape to snapshot matrix
        X = reshape(field, [-1, self.Ntime])
        # make an singular value decomposition of the snapshot matrix
        # and reduce it to the specified numer of moddes
        [U, S, VT] = self.reduce(X, self.Nmodes)
        # the snapshot matrix is only stored with reduced number of SVD modes
        self.modal_system = {"U": U, "sigma": S, "VT": VT}

    def build_field(self):
        """
        Calculate the field from the SVD modes: X=U*S*VT
        """
        # modes from the singular value decomposition
        u = self.modal_system["U"]
        s = self.modal_system["sigma"]
        vh = self.modal_system["VT"]
        # add up all the modes A=U * S * VH
        return np.reshape(np.dot(u * s, vh), self.data_shape)

    def plot_field(self, field_index=0):
        
        
        if self.dim == 1:
            snapshot = self.inv_shift(self.build_field()[field_index])
            # the snapshot matrix is transposed 
            # in order to have the time on the y axis
            pcolor(snapshot.T)
            #plt.xlabel(r"$N_x$")
            #plt.ylabel(r"$N_t$")

        if self.dim == 2:
            qframe = self.build_field()
            plt.pcolormesh(qframe[...,0,-1])
        

    def plot_singular_values(self):
        """
        This function plots the singular values of the frame.
        """
        sigmas = self.modal_system["sigma"]
        semilogy(sigmas/sigmas[0], "r+")
        xlabel(r"$i$")
        ylabel(r"$\sigma_i/\sigma_0$")

    def concatenate(self, other):
        """ Add two frames for the purpose of concatenating there modes """
        # TODO make check if other and self can be added:
        # are they in the same frame? Are they from the same data etc.
        new = frame(self.trafo,
                    self.build_field(), self.Nmodes)
        # combine left singular vecotrs
        Uself = self.modal_system["U"]
        Uother = other.modal_system["U"]
        new.modal_system["U"] = np.concatenate([Uself, Uother], axis=1)

        # combine right singular vecotrs
        VTself = self.modal_system["VT"]
        VTother = other.modal_system["VT"]
        new.modal_system["VT"] = np.concatenate([VTself, VTother], axis=0)

        Sself = self.modal_system["sigma"]
        Sother = other.modal_system["sigma"]
        new.modal_system["sigma"] = np.concatenate([Sself, Sother])

        new.Nmodes += other.Nmodes

        return new
    
    def __add__(self, other):
        """ Add two frames  """
        if isinstance(other,frame):    
            new_field = self.build_field() + other.build_field()
        elif np.shape(other)==self.data_shape:
            new_field = self.build_field() + other
        
        # apply svd and save modes
        self.set_orhonormal_system(new_field)

        return self

# %%
###############################################################################
# Determination of shift velocities
###############################################################################

def shift_velocities(dx, dt, fields, n_velocities, v_min, v_max, v_step, n_modes):
    sigmas = np.zeros([int((v_max-v_min)/v_step), n_modes])
    v_shifts = np.linspace(v_min, v_max, int((v_max-v_min)/v_step))

    i = 0
    for v in v_shifts:
        example_frame = frame(v, dx, dt, fields, n_modes)
        sigmas[i, :] = example_frame.modal_system["sigma"]
        i += 1

    # Plot singular value spectrum
    plt.plot(v_shifts, sigmas, 'o')

    sigmas_temp = sigmas.copy()
    c_shifts = []

    for i in range(n_velocities):
        max_index = np.where(sigmas_temp == sigmas_temp.max())
        max_index_x = max_index[0]
        max_index_x = max_index_x[0]
        max_index_y = max_index[1]
        max_index_y = max_index_y[0]

        sigmas_temp[max_index_x, max_index_y] = 0

        c_shifts.append(v_shifts[max_index_x])

    return c_shifts


###############################################################################
# least square minimization
###############################################################################

def minimize(Xtilde_frames, X):
    ######################################################
    # build coef matrix X_coef_mat of eq. (9) in Reiss2017
    ######################################################
    X_coef = []        # lab frame
    X_coef_shift = []  # co-moving frame
    # loop through all moving frames
    for k, frame in enumerate(Xtilde_frames):
        # left singular vectors
        U = frame.modal_system["U"]
        # right singular vectors
        VT = frame.modal_system["VT"]
        # singular values
        Nmodes = frame.Nmodes
        # loop over all the modes
        # each mode represents one alpha_k and one column of the matrix X
        for i in range(Nmodes):
            Xmode_shift = np.outer(U[:, i], VT[i, :])
            Xmode = frame.inv_shift(Xmode_shift)

            X_coef_shift.append(reshape(Xmode_shift, [-1]))
            X_coef.append(reshape(Xmode, [-1]))
    ######################################################
    # solve the minimization problem
    ######################################################
    # first we convert the list of vectors (reshaped matrices) to an array.
    # We have to transpose it since  the function asarray
    # concatenates the list vectors verticaly
    X_coef = asarray(X_coef).T
    X_coef_shift = asarray(X_coef_shift).T
    X_ref = reshape(X.copy(), [-1, 1])
    # X_coef * alpha = X
    alpha = lstsq(X_coef, X_ref)[0]
    alpha = asarray(alpha)

    return alpha, X_coef_shift


###############################################################################
# update the Xtilde frames and truncate modes
###############################################################################

def update_and_reduce_modes(Xtilde_frames, alpha, X_coef_shift, Nmodes_reduce):

    """
    This function implements the 5. step of the SPOD algorithm (see Reiss2017)
    - calculate the new modes from the optimiced alpha combining 
    Xtilde and R modes.
    - truncate the number of modes to the desired number of reduced modes 
    - update the new Xtilde in the corresponding frames
    """
    for k, frame in enumerate(Xtilde_frames):
        Nmodes = frame.Nmodes
        alpha_k = alpha[k*Nmodes:(k+1)*Nmodes]
        # linear combination to get the new Xtilde
        Xnew_k = X_coef_shift[:, k*Nmodes:(k+1)*Nmodes] @ alpha_k
        Xnew_k = reshape(Xnew_k, [-1, frame.Ntime])
        frame.Nmodes = Nmodes_reduce  # reduce to the desired number of modes
        [U, S, VT] = frame.reduce(Xnew_k, self.Nmodes)
        frame.modal_system = {"U": U, "sigma": S, "VT": VT}

# %%
###############################################################################
# sPOD algorithm
###############################################################################

  
        
        
###############################################################################
# distribute the residual of frame
###############################################################################        
def sPOD_distribute_residual(q, transforms, nmodes, eps, Niter=1, visualize=True):     
    #########################    
    ## 1.Step: Initialize
    #########################
    qtilde = np.zeros_like(q)
    qtilde_frames = [frame(trafo, qtilde, number_of_modes=nmodes) for trafo in transforms]
    norm_q = norm(reshape(q,-1))
    Nframes = len(transforms) 
    it = 0
    
    rel_err = 1
    while rel_err > eps and it < Niter:

        it += 1  # counts the number of iterations in the loop
        #############################
        # 2.Step: Calculate Residual
        #############################
        res = q - qtilde
        norm_res = norm(reshape(res,-1))
        rel_err = norm_res/norm_q
        qtilde = np.zeros_like(q)
        print( "it=%d rel_err= %4.4e"%(it,rel_err))
        ###########################
        # 3. Step: update frames
        ##########################
        t = time.time()
        for k, (trafo,q_frame) in enumerate(zip(transforms,qtilde_frames)):
            #R_frame = frame(trafo, res, number_of_modes=nmodes)
            res_shifted = trafo.apply(res)
            q_frame_field = q_frame.build_field()
            q_frame.set_orhonormal_system(q_frame_field + res_shifted/Nframes)
            #q_frame += R_frame.build_field()/Nframes
            qtilde += trafo.reverse(q_frame.build_field())
        elapsed = time.time() - t
        print( "elapsed time=%d"%(elapsed))

    return qtilde_frames, qtilde
        
        
        
        
###############################################################################
# shift and reduce
###############################################################################        
def sPOD_shift_and_reduce(X, n_velocities, dx, dt, nmodes=2, eps=1e-4, Niter=5, visualize=True):
    """
    First version Shift&Reduce of the shifted POD algorithm published in
    Reiss2018: https://arxiv.org/abs/1512.01985
    """
    
    # Determine shift velocities
    velocities = shift_velocities(dx, dt, X, n_velocities,
                                  v_min=-5, v_max=5, v_step=0.01, n_modes=1)

    
    # plot the first component of the original field
    if visualize:
        subplot(1, 4, 1)
        p = pcolor(X[0].T)
        plt.title(r'$X$')
        plt.ylabel(r"$N_t$")
        plt.xlabel(r"$N_x$")
        plt.pause(0.05)
        clims = p.get_clim()
    
    #################################
    # 1. reset loop variables
    ################################

    Xtilde = np.zeros(np.shape(X))
    Xtilde_frames = [frame(v, dx, dt, Xtilde, nmodes) for v in velocities]
    rel_err = 1
    it = 0
    results = {"rel_err": []} # save all the output results in the dict
    ###########################################################################
    # MAIN LOOP
    ###########################################################################
    # loop until the desired precission is achieved or the maximal number
    # of iterations
    while rel_err > eps and it < Niter:

        it += 1  # counts the number of iterations in the loop

        ###############################
        # 2. calculate residual R
        ###############################
        R = X - Xtilde
        rel_err = norm(R)/norm(X)  # relative error
        results["rel_err"].append(rel_err)
        ##########################################
        print("iter= ", it, "rel err= ", rel_err, "elapsed time=", time.time() - t)
        ##########################################
        t = time.time()
        ###############################
        # 3. Multi shift and reduce R
        ###############################
        R_frames = [frame(v, dx, dt, R, nmodes) for v in velocities]

        if visualize:  # plot the residual
            subplot(1, len(Xtilde_frames)+2, 2)
            pcolor(R[0].T)
            plt.title(r"$R=X-\tilde{X}$")
            # plt.xlabel(r"$N_x$")
            plt.pause(0.05)

        #################################
        # 4. optimize with least squares
        #################################
        #######################################################################
        # a) combine the modes of Xtilde and R
        # Note 2 objects in the same frame, can be added by concatenating
        # the SVD left and right singular vectors.
        #######################################################################
        # Xtilde_frames = []
        for k, R_frame in enumerate(R_frames):
            Xtilde_frames[k] = Xtilde_frames[k].concatenate(R_frame)
        ###################################################
        # b) Solve (Xtilde+R) * alpha = X for unknown alpha
        #  + the classical method (Reiss2018) uses the
        #    least square approach
        # TODO Implement the other methods as well
        # TODO why is it not converging faster???
        ###################################################
        [alpha, Xhat_coef] = minimize(Xtilde_frames, X)
        ######################################################
        # 5. calculate new modes from optimized coefficients
        ######################################################
        update_and_reduce_modes(Xtilde_frames, alpha, Xhat_coef, nmodes)
        #############################################
        # 6. update approximation:
        # Add up all the frames to compute Xtilde
        #############################################
        Xtilde *= 0
        for k, Xframe in enumerate(Xtilde_frames):

            Xtilde += Xframe.inv_shift(Xframe.build_field()[0])
            # we plot the first 3 frames
            if visualize and k <= 3:
                subplot(1, len(Xtilde_frames)+2, k+3)
                Xframe.plot_field()
                plt.clim(clims)
                plt.title(r"$q^"+str(k)+"(x,t)$")
                # plt.xlabel(r"$N_x$")
            plt.pause(0.05)

    ###########################################################################
    # End of MAIN LOOP
    ###########################################################################
    return Xtilde_frames, results.get('rel_err')



    
        

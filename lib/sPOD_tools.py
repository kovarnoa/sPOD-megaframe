#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:19:19 2018

@author: Philipp Krah

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

    def __init__(self, transform, field=None, number_of_modes=None):
        """
        Initialize a co moving frame.
        """
        data_shape = transform.data_shape
        self.data_shape = data_shape
        self.Ngrid = np.prod(data_shape[:3])
        self.Ntime = data_shape[3]
        self.trafo = transform
        self.dim = self.trafo.dim
        if not number_of_modes:
            self.Nmodes = self.Ntime
        else:
            self.Nmodes = number_of_modes
        # transform the field to reference frame
        if not np.all(field == None):
            field = self.trafo.apply(field)
            self.set_orthonormal_system(field)
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
        
    def set_orthonormal_system(self, field):
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
        return np.dot(u * s, vh)

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
        self.set_orthonormal_system(new_field)

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
# Proximal operators
###############################################################################
def shrink(X, tau):
    """
    Proximal Operator for 1 norm minimization

    :param X: input matrix or vector
    :param tau: threshold
    :return: argmin_X1 tau * || X1 ||_1 + 1/2|| X1 - X||_F^2
    """
    return np.sign(X)*np.maximum(np.abs(X)-tau, 0)

def SVT(X, mu):
    """
    Proximal Operator for schatten 1 norm minimization
    :param X: input matrix for thresholding
    :param mu: threshold
    :return: argmin_X1 mu|| X1 ||_* + 1/2|| X1 - X||_F^2
    """
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    s = shrink(s, mu)
    return (u, s, vt)

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
        [U, S, VT] = frame.reduce(Xnew_k, Nmodes_reduce)
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
    Nframes = len(transforms)
    if np.size(nmodes) != Nframes:
        nmodes = list([nmodes]) * Nframes
    qtilde_frames = [frame(trafo, qtilde, number_of_modes=nmodes[k]) for k,trafo in enumerate(transforms)]
    norm_q = norm(reshape(q,-1))
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

        ###########################
        # 3. Step: update frames
        ##########################
        t = time.time()
        for k, (trafo,q_frame) in enumerate(zip(transforms,qtilde_frames)):
            #R_frame = frame(trafo, res, number_of_modes=nmodes)
            res_shifted = trafo.apply(res)
            q_frame_field = q_frame.build_field()
            q_frame.set_orthonormal_system(q_frame_field + res_shifted/Nframes)
            #q_frame += R_frame.build_field()/Nframes
            qtilde += trafo.reverse(q_frame.build_field())
        elapsed = time.time() - t
        print("it=%d rel_err= %4.4e t_cpu = %2.2f" % (it, rel_err, elapsed))

    return qtilde_frames, qtilde


###############################################################################
# shifted rPCA
###############################################################################
# def shifted_rPCA_(snapshot_matrix, transforms, eps, Niter=1, visualize=True):
#     """
#     Currently this method doesnt work. Its very similar to distribute residual, but does not seem to converge
#     :param snapshot_matrix: M x N matrix with N beeing the number of snapshots
#     :param transforms: Transformations
#     :param eps: stopping criteria
#     :param Niter: maximal number of iterations
#     :param visualize: if true: show intermediet results
#     :return:
#     """
#     assert (np.ndim(snapshot_matrix) == 2), "Are you stephen hawking trying to solve this problem in 16 dimensions?" \
#                              "Please give me a snapshotmatrix with every snapshot in one column"
#     #########################
#     ## 1.Step: Initialize
#     #########################
#     qtilde = np.zeros_like(snapshot_matrix)
#     E = np.zeros_like(snapshot_matrix)
#     Y = np.zeros_like(snapshot_matrix)
#     qtilde_frames = [frame(trafo, qtilde) for trafo in transforms]
#     q = snapshot_matrix.copy()
#     norm_q = norm(reshape(q, -1))
#     it = 0
#     M, N = np.shape(q)
#     mu = M * N / (4 * np.sum(np.abs(q)))
#     lambd = 10 * 1 / np.sqrt(np.maximum(M, N))
#     thresh = 1e-7 * norm_q
#     mu_inv = 1 / mu
#     rel_err = 1
#     res = q # in the first step the residuum is q since qtilde is 0
#     while rel_err > eps and it < Niter:
#
#         it += 1  # counts the number of iterations in the loop
#         #############################
#         # 2.Step: set qtilde to 0
#         #############################
#         qtilde = np.zeros_like(q)
#         ranks = []
#         ###########################
#         # 3. Step: update frames
#         ##########################
#         t = time.time()
#         for k, (trafo, q_frame) in enumerate(zip(transforms, qtilde_frames)):
#             # R_frame = frame(trafo, res, number_of_modes=nmodes)
#             q_frame_field = q_frame.build_field()
#             res_shifted = trafo.apply(res + mu_inv * Y)
#             q_frame_field += res_shifted
#             [U, S, VT] = SVT(q_frame_field, mu_inv)
#             rank = np.sum(S > 0)
#             q_frame.modal_system = {"U": U[:,:rank], "sigma": S[:rank], "VT": VT[:rank,:]}
#             # q_frame += R_frame.build_field()/Nframes
#             qtilde += trafo.reverse(q_frame.build_field())
#             ranks.append(rank) # list of ranks for each frame
#         ###########################
#         # 4. Step: update noice term
#         ##########################
#         E = shrink(q - qtilde + mu_inv * Y, lambd * mu_inv)
#         #############################
#         # 5. Step: update multiplier
#         #############################
#         res = q - qtilde - E
#         Y = Y + mu * (res)
#
#         norm_res = norm(reshape(res, -1))
#         rel_err = norm_res / norm_q
#
#         elapsed = time.time() - t
#         print("it=%d rel_err= %4.4e norm(E) = %4.1e tcpu = %2.2f, ranks_frame = " % (it, rel_err, norm(reshape(E, -1)), elapsed), *ranks)
#
#     return qtilde_frames, qtilde



###############################################################################
# shifted rPCA
###############################################################################
def shifted_rPCA(snapshot_matrix, transforms, eps, Niter=1, visualize=True):
    """
    :param snapshot_matrix: M x N matrix with N beeing the number of snapshots
    :param transforms: Transformations
    :param eps: stopping criteria
    :param Niter: maximal number of iterations
    :param visualize: if true: show intermediet results
    :return:
    """
    assert (np.ndim(snapshot_matrix) == 2), "Are you stephen hawking trying to solve this problem in 16 dimensions?" \
                             "Please give me a snapshotmatrix with every snapshot in one column"
    #########################
    ## 1.Step: Initialize
    #########################
    qtilde = np.zeros_like(snapshot_matrix)
    E = np.zeros_like(snapshot_matrix)
    Y = np.zeros_like(snapshot_matrix)
    qtilde_frames = [frame(trafo, qtilde) for trafo in transforms]
    q = snapshot_matrix.copy()
    norm_q = norm(reshape(q, -1))
    it = 0
    M, N = np.shape(q)
    mu = M * N / (4 * np.sum(np.abs(q)))*0.001
    lambd = 10 * 1 / np.sqrt(np.maximum(M, N))
    thresh = 1e-7 * norm_q
    mu_inv = 1 / mu
    rel_err = 1
    while rel_err > eps and it < Niter:

        it += 1  # counts the number of iterations in the loop
        #############################
        # 2.Step: set qtilde to 0
        #############################
        qtilde = np.zeros_like(q)
        ranks = []
        ###########################
        # 3. Step: update frames
        ##########################
        t = time.time()
        for k, (trafo, q_frame) in enumerate(zip(transforms, qtilde_frames)):
            qtemp = 0
            for p, (trafo_p, frame_p) in enumerate(zip(transforms, qtilde_frames)):
                if p != k:
                    qtemp += trafo_p.reverse(frame_p.build_field())

            qk = trafo.apply(q - qtemp - E + mu_inv * Y)
            [U, S, VT] = SVT(qk, mu_inv)
            rank = np.sum(S > 0) + 1
            q_frame.modal_system = {"U": U[:,:rank], "sigma": S[:rank], "VT": VT[:rank,:]}
            ranks.append(rank) # list of ranks for each frame
            qtilde += trafo.reverse(q_frame.build_field())
        ###########################
        # 4. Step: update noice term
        ##########################
        E = shrink(q - qtilde + mu_inv * Y, lambd * mu_inv)
        #############################
        # 5. Step: update multiplier
        #############################
        res = q - qtilde - E
        Y = Y + mu * (res)

        norm_res = norm(reshape(res, -1))
        rel_err = norm_res / norm_q

        elapsed = time.time() - t
        print("it=%d rel_err= %4.4e norm(E) = %4.1e tcpu = %2.2f, ranks_frame = " % (
        it, rel_err, norm(reshape(E, -1)), elapsed), *ranks)

    return qtilde_frames, qtilde


###############################################################################
# shift and reduce
###############################################################################        
def sPOD_shift_and_reduce(X, n_velocities, dx, dt, nmodes=2, eps=1e-4, Niter=5, visualize=True):
    """
    First version Shift&Reduce of the
    shifted POD algorithm published in
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



    
        

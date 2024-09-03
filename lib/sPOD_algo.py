# -*- coding: utf-8 -*-
"""
Created on Wed Mar 06 15:08:42 2024

@author: Philipp Krah, Beata Zorawski, Arthur Marmin

This file provides the algorithms that solve the optimization problem associated
with the robust shifter proper orthogonal decomposition (sPOD).
"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
import os
import time

import numpy as np
from numpy import reshape
from numpy.linalg import norm
from dataclasses import dataclass
from warnings import warn
from sPOD_tools import Frame, SVT, trunc_svd, trunc_SVT, shrink
from total_variation import solve_TVL1, solve_ROF
from transforms import Transform

# ============================================================================ #


# ============================================================================ #
#                            CLASS of Return Values                            #
# ============================================================================ #
class ReturnValue:
    """
    This class inherits all return values of the shifted POD routines
    """

    def __init__(
        self,
        frames,
        approximation,
        relative_error_hist=None,
        ranks=None,
        ranks_hist=None,
        error_matrix=None,
        stat_frame = None
    ):
        """
        Constructor.

        :param frames:
        :type frames:

        :param approximation:
        :type approximation:

        :param relative_err_hist:
        :type relative_err_hist: , optional

        :param ranks:
        :type ranks: , optional

        :param ranks_hist:
        :type ranks_hist: , optional

        :param error_matrix:
        :type error_matrix: , optional
        """
        self.frames = frames  # List of all frames
        self.data_approx = approximation  # Approximation of the snapshot data
        if relative_error_hist is not None:
            self.rel_err_hist = relative_error_hist
        if error_matrix is not None:
            self.error_matrix = error_matrix
        if ranks is not None:
            self.ranks = ranks
        if ranks_hist is not None:
            self.ranks_hist = ranks_hist
        if stat_frame is not None:
            self.stat_frame = stat_frame


# ============================================================================ #


# ============================================================================ #
#                                sPOD ALGORITHMS                               #
# ============================================================================ #
def shifted_POD(snapshot_matrix, transforms, myparams, method, param_alm=None, nmodes=None, qt_frames=None, nmodesstat=0):
    """
    This function aggregates all the different shifted_POD_Algo() methods to
    provide a unique interface.

    :param snapshot_matrix: Snapshot matrix with with dimensions :math:`M \times N`,
                            :math:`N` is the number of snapshots (i.e. time stamps)
                            and :math:`M` is the number of number of spatial samples
                            (i.e. the ODE dimension).
    :type snapshot_matrix: :class:`numpy.ndarray` (2-dimensional)

    :param transforms: List of transformations associated with the co-moving fields.
    :type transforms: list[Transform]

    :param nmodes: Number of modes to use in each frame
    :type nmodes: integer

    :param myparams: Parameters for the JFB algorithm
    :type myparams: sPOD_Param

    :param method: Name of the method
    :type myparams: string

    :param param_alm: Parameter mu for ALM algorithm
    :type myparams: float

    :return:
    :rtype: :class:`ReturnValue`
    """
    if method == "ALM":
        return shifted_POD_ALM(
            snapshot_matrix,
            transforms,
            myparams,
            nmodes_max=nmodes,
            mu=param_alm,
            qt_frames=qt_frames
        )
    elif method == "BFB":
        return shifted_POD_FB(
            snapshot_matrix, transforms, myparams, nmodes_max=nmodes, method="BFB"
        )
    elif method == "JFB":
        return shifted_POD_FB(
            snapshot_matrix, transforms, myparams, nmodes_max=nmodes, method="JFB"
        )
    elif method == "J2":
        return shifted_POD_J2(snapshot_matrix, transforms, nmodes, myparams)
    
    elif method == "J2_megaframe":
        return shifted_POD_J2_megaframe(snapshot_matrix, transforms, myparams, nmodes, nmodesstat=nmodesstat)
    
    elif method == "BFB_megaframe":
        return shifted_POD_FB_megaframe(snapshot_matrix, transforms, myparams, nmodes_max=nmodes, method="BFB")
        
    elif method == "JFB_megaframe":
        return shifted_POD_FB_megaframe(snapshot_matrix, transforms, myparams, nmodes_max=nmodes, method="JFB")
        
    elif method == "ALM_megaframe":
        return shifted_POD_ALM_megaframe(snapshot_matrix, transforms, myparams, nmodes_max=nmodes, mu=param_alm)

def shifted_POD_J2(
    snapshot_matrix,
    transforms,
    nmodes,
    myparams,
):
    """
    This function implements the J2 algorithm.

    :param snapshot_matrix: Snapshot matrix with with dimensions :math:`M \times N`,
                            :math:`N` is the number of snapshots (i.e. time stamps)
                            and :math:`M` is the number of number of spatial samples
                            (i.e. the ODE dimension).
    :type snapshot_matrix: :class:`numpy.ndarray` (2-dimensional)

    :param transforms: List of transformations associated with the co-moving fields.
    :type transforms: list[Transform]

    :param nmodes: Number of modes to use in each frame
    :type nmodes: integer

    :param myparams: Parameters for the JFB algorithm
    :type myparams: sPOD_Param

    :return:
    :rtype: :class:`ReturnValue`
    """

    assert (
        np.ndim(snapshot_matrix) == 2
    ), "Please give enter a snapshotmatrix with every snapshot in one column"
    if myparams.use_rSVD:
        warn(
            "Warning: Using rSVD to accelarate decomposition procedure may lead "
            "to different results."
        )

    ###################################
    #        1. Initialization        #
    ###################################
    q = snapshot_matrix
    qtilde = np.zeros_like(q)
    nTransports = len(transforms)
    if np.size(nmodes) != nTransports:
        nmodes = list([nmodes]) * nTransports
    qtilde_frames = [
        Frame(transfo, qtilde, Nmodes=nmodes[k]) for k, transfo in enumerate(transforms)
    ]
    norm_q = norm(reshape(q, -1))

    ###########################
    # Error of the truncated SVD
    r_ = np.sum(nmodes)
    (u, s, vt) = trunc_svd(q, nmodes_max=r_, use_rSVD=myparams.use_rSVD)
    err_svd = np.linalg.norm(q - np.dot(u * s, vt), ord="fro") / norm_q
    print(
        "Relative error using a truncated SVD with {:d} modes:{:4.4e}".format(
            r_, err_svd
        )
    )
    ###########################

    current_it = 0
    rel_err = 1
    rel_err_list = [1.0]
    ranks_hist = [[] for r in range(nTransports)]
    sum_elapsed = 0
    while rel_err > myparams.eps and current_it < myparams.maxit:
        current_it += 1
        #############################
        # 2.Step: Calculate Residual
        #############################
        res = q - qtilde
        qtilde = np.zeros_like(q)
        ranks = [0] * nTransports

        ###########################
        # 3. Step: update frames
        ##########################
        t = time.perf_counter()
        for k, (trafo, q_frame) in enumerate(zip(transforms, qtilde_frames)):
            res_shifted = trafo.reverse(res)
            q_frame_field = q_frame.build_field()
            stepsize = 1 / nTransports
            q_frame.set_orthonormal_system(
                q_frame_field + stepsize * res_shifted, myparams.use_rSVD
            )

            if myparams.total_variation_iterations > 0:
                q_frame.smoothen_time_amplitudes(
                    TV_iterations=myparams.total_variation_iterations, 
                    clambda=myparams.tv_lambda
                )
            qtilde += trafo.apply(q_frame.build_field())
            S = q_frame.modal_system["sigma"]
            U = q_frame.modal_system["U"]
            VT = q_frame.modal_system["VT"]
            rank = np.sum(S > 0)
            ranks[k] = rank
            ranks_hist[k].append(rank)
        
        # rel errors etc.
        norm_res = norm(reshape(res, -1))
        rel_err = norm_res / norm_q
        rel_err_list.append(rel_err)
        elapsed = time.perf_counter() - t
        sum_elapsed += elapsed
        
        print(
            "it=%d rel_err= %4.4e t_cpu = %2.2f, ranks_frame ="
            % (current_it, rel_err, elapsed),
            *ranks
        )
        if (current_it > 5) and (
            np.abs(rel_err_list[-1] - rel_err_list[-4])
            < myparams.gtol * abs(rel_err_list[-1])
        ):
            break
    print("CPU time in total: ", sum_elapsed)
    return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist)


def shifted_POD_FB(
    snapshot_matrix,
    transforms,
    myparams,
    nmodes_max=None,
    method="BFB",
):
    """
    This function implements the Forward-Backward method (FB).

    :param snapshot_matrix: Snapshot matrix with with dimensions :math:`M \times N`,
                            :math:`N` is the number of snapshots (i.e. time stamps)
                            and :math:`M` is the number of number of spatial samples
                            (i.e. the ODE dimension).
    :type snapshot_matrix: :class:`numpy.ndarray` (2-dimensional)

    :param transforms: List of transformations associated with the co-moving fields.
    :type transforms: list[Transform]

    :param nmodes: Number of modes to use in each frame
    :type nmodes: integer

    :param myparams: Parameters for the FB algorithm
    :type myparams: class:`sPOD_Param`

    :param method: Choice of the version of FB. Options are "BFB" (Block-coordinate
                   Forward Backward) and "JFB" (Joint Forward Backward).

    :return:
    :rtype: :class:`ReturnValue`
    """

    assert (
        np.ndim(snapshot_matrix) == 2
    ), "Please give enter a snapshotmatrix with every snapshot in one column"
    if myparams.use_rSVD:
        warn(
            "Warning: Using rSVD to accelarate decomposition procedure may lead "
            "to different results."
        )
        
    print('lambda_s = %f'%myparams.lambda_s)
    print('lambda_e = %f'%myparams.lambda_E)
    print(myparams.isError)

    ###################################
    #        1. Initialization        #
    ###################################
    q = snapshot_matrix
    qtilde = np.zeros_like(q)
    if myparams.isError:
        E = np.zeros_like(snapshot_matrix)
    nTransports = len(transforms)
    # make a list of the number of maximal ranks in each frame
    if not np.all(nmodes_max):  # check if array is None, if so set nmodes_max onto N
        nmodes_max = np.min(np.shape(snapshot_matrix)) # use the smallest dimension beacuse after this singular values will be 0
    if np.size(nmodes_max) != nTransports:
        nmodes = list([nmodes_max]) * nTransports
    else:
        nmodes = nmodes_max
    qtilde_frames = [
        Frame(transfo, qtilde, Nmodes=nmodes[k]) for k, transfo in enumerate(transforms)
    ]
    norm_q = norm(reshape(q, -1))

    ###########################
    # Error of the truncated SVD
    r_ = np.sum(nmodes)
    (u, s, vt) = trunc_svd(q, nmodes_max=None, use_rSVD=myparams.use_rSVD)
    err_svd = np.linalg.norm(q - np.dot(u * s, vt), ord="fro") / norm_q
    print(
        "Relative error using a truncated SVD with {:d} modes:{:4.4e}".format(
            r_, err_svd
        )
    )
    ###########################

    current_it = 0
    objective_0 = 0.5 * norm(q, ord="fro") ** 2
    objective_list = [objective_0]
    rel_decrease = 1
    rel_decrease_list = [1]
    rel_err_list = [1.0]
    ranks_hist = [[] for r in range(nTransports)]
    sum_elapsed = 0
           
    while rel_err > myparams.eps and current_it < myparams.maxit:
        current_it += 1
        ###################################
        #      2. Calculate residual      #
        ###################################
        if myparams.isError:
            res = q - qtilde - E
        else:
            res = q - qtilde

        if method == "JFB":
            qtilde = np.zeros_like(q)
        ranks = [0] * nTransports

        ###################################
        #      3. Update the frames       #
        ###################################
        t = time.perf_counter()
        for k, (trafo, q_frame) in enumerate(zip(transforms, qtilde_frames)):
            res_shifted = trafo.reverse(res)
            q_frame_field = q_frame.build_field()
            if method == "BFB":
                qtilde -= trafo.apply(q_frame.build_field())
            stepsize = 1 / nTransports
            q_frame.set_orthonormal_system_svt(
                q_frame_field + stepsize * res_shifted, stepsize * myparams.lambda_s
            )

            if myparams.total_variation_iterations > 0:
                q_frame.smoothen_time_amplitudes(
                    TV_iterations=myparams.total_variation_iterations, 
                    clambda=myparams.tv_lambda
                )

            S = q_frame.modal_system["sigma"]
            rank = np.sum(S > 0)
            ranks[k] = rank
            ranks_hist[k].append(rank)
            qtilde += trafo.apply(q_frame.build_field())
            if method == "BFB":
                if myparams.isError:
                    res = q - qtilde - E
                else:
                    res = q - qtilde
        if myparams.isError:
            E = shrink(E + stepsize * res, stepsize * myparams.lambda_E)
            objective = (
                0.5 * norm(res, ord="fro") ** 2
                + myparams.lambda_s
                * sum(norm(qk.build_field(), ord="nuc") for qk in qtilde_frames)
                + myparams.lambda_E * norm(E, ord=1)
            )
        else:
            objective = 0.5 * norm(res, ord="fro") ** 2 + myparams.lambda_s * sum(
                norm(qk.build_field(), ord="nuc") for qk in qtilde_frames
            )
        objective_list.append(objective)
        rel_decrease = np.abs((objective_list[-1] - objective_list[-2])) / np.abs(
            objective_list[-1]
        )

        rel_decrease_list.append(rel_decrease)
        
        norm_res = norm(reshape(res, -1))
        rel_err = norm_res / norm_q
        rel_err_list.append(rel_err)
        
        elapsed = time.perf_counter() - t
        sum_elapsed += elapsed
        print(rel_decrease)
        if myparams.isVerbose:
            print(
                "Iter {:4d} / {:d} | Rel_err= {:4.4e} | t_cpu = {:2.2f}s | "
                "ranks_frame = ".format(current_it, myparams.maxit, rel_err, elapsed),
                *ranks
            )
        if (current_it > 5) and (rel_decrease < myparams.gtol):
            break

    if myparams.isError:
        if myparams.isVerbose:
            print("CPU time in total: ", sum_elapsed)
        return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist, E)
    if myparams.isVerbose:
        print("CPU time in total: ", sum_elapsed)
    return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist)


def shifted_POD_ALM(snapshot_matrix, transforms, myparams, nmodes_max=None, mu=None, qt_frames=None):
    """
    This function implements the Augmented Lagangian method (ALM).

    :param snapshot_matrix: Snapshot matrix with with dimensions :math:`M \times N`,
                            :math:`N` is the number of snapshots (i.e. time stamps)
                            and :math:`M` is the number of number of spatial samples
                            (i.e. the ODE dimension).
    :type snapshot_matrix: :class:`numpy.ndarray` (2-dimensional)

    :param transforms: List of transformations associated with the co-moving fields.
    :type transforms: list[Transform]

    :param myparams: Parameters for ALM algorithm
    :type myparams: class:`sPOD_Param`

    :param nmodes_max: Maximal number of modes allowed in each frame, default is
                       the number of snapshots :math:`N`.
                       Note: it is good to give a number large enough in order to
                       get the error down but smaller than :math:`N`.
                       This will increase the performance of the algorith.

    :param mu: Parameter of the augmented Lagrangian (i.e. weight of the
               quadratic term).
    :type mu: float

    :return:
    :rtype: :class:`ReturnValue`
    """
    # Note (AK): the implementation is slightly modified compared to the sPOD-proxbranch, but produces exactly the same results
    #         sPOD-proxbranch needs (nTransports^2 + nTransports) shifts per iter, this one only (3*nTransports)
    #         this one is faster for nTransports>2
    #         sPOD-proxbranch is still faster for nTransports=2
    assert (
        np.ndim(snapshot_matrix) == 2
    ), "Please give enter a snapshotmatrix with every snapshot in one column"
    if myparams.use_rSVD:
        warn(
            "Warning: Using rSVD to accelarate decomposition procedure may lead "
            "to different results."
        )
    ###################################
    #        1. Initialization        #
    ###################################
    qtilde = np.zeros_like(snapshot_matrix)
    E = np.zeros_like(snapshot_matrix)
    nTransports = len(transforms)

    # make a list of the number of maximal ranks in each frame
    if not np.all(nmodes_max):  # check if array is None, if so set nmodes_max onto N
        nmodes_max = np.min(np.shape(snapshot_matrix)) # use the smallest dimension beacuse after this singular values will be 0
    if np.size(nmodes_max) != nTransports:
        nmodes = list([nmodes_max]) * nTransports
    else:
        nmodes = nmodes_max
     
    if qt_frames is None:
            qtilde_frames = [Frame(transfo, field=qtilde, Nmodes=nmodes[k]) for k, transfo in enumerate(transforms)]
    else:
            qtilde_frames = [Frame(transfo, field=qt_frames[k], Nmodes=nmodes[k]) for k, transfo in enumerate(transforms)]
    if myparams.isVerbose:
        print('lambda_s = %f'%myparams.lambda_s)
        print('lambda_e = %f'%myparams.lambda_E)
        print('mu = %f'%mu)

    q = snapshot_matrix.copy()
    Y = np.zeros_like(snapshot_matrix)
    norm_q = norm(reshape(q, -1))
    it = 0
    mu_inv = 1 / mu
    rel_err = 1
    res_old = 0
    rel_err_list = []
    ranks_hist = [[] for r in range(nTransports)]
    sum_elapsed = 0
    qtilde = np.zeros_like(q)
    while rel_err > myparams.eps and it < myparams.maxit:
        it += 1  # counts the number of iterations in the loop
        ###################################
        #       2. Set qtilde to 0        #
        ###################################
        #qtilde = np.zeros_like(q)
        ranks = [0] * nTransports
        ###################################
        #      3. Update the frames       #
        ###################################
        t = time.perf_counter()

        for k, (trafo, q_frame) in enumerate(zip(transforms, qtilde_frames)):
            qk = q_frame.build_field() + trafo.reverse(q - qtilde - E + mu_inv * Y) 
            
            qtilde -= trafo.apply(q_frame.build_field()) 
            
            q_frame.set_orthonormal_system_svt(qk, mu_inv * myparams.lambda_s) 
            rank = len(q_frame.modal_system["sigma"])
            
            if (myparams.total_variation_iterations > 0) and (rank > 0):
                q_frame.smoothen_time_amplitudes(
                    TV_iterations=myparams.total_variation_iterations, clambda=myparams.tv_lambda
                ) 
                                  
            #[U, S, VT] = SVT(qk, mu_inv * myparams.lambda_s, q_frame.Nmodes, myparams.use_rSVD)
            #rank = np.sum(S > 0)
            #q_frame.modal_system = {
            #    "U": U[:, :rank],
            #    "sigma": S[:rank],
            #    "VT": VT[:rank, :],
            #}
            ranks_hist[k].append(rank)
            ranks[k] = rank

            qtilde += trafo.apply(q_frame.build_field())

        ###################################
        #    4. Update the noise term     #
        ###################################
        if myparams.isError:
            E = shrink(q - qtilde + mu_inv * Y, myparams.lambda_E * mu_inv)
        ###################################
        #      5. Update multipliers      #
        ###################################
        if myparams.isError:
            res = q - qtilde - E
        else:
            res = q - qtilde
        Y = Y + mu * res

        ###################################
        #          6. Update mu           #
        ###################################
        dres = norm(res, ord="fro") - res_old
        res_old = norm(res, ord="fro")
        norm_dres = np.abs(dres)

        norm_res = norm(reshape(res, -1))
        rel_err_without_noise = norm(reshape(res + E, -1)) / norm_q
        rel_err = norm_res / norm_q
        rel_err_list.append(rel_err)
        elapsed = time.perf_counter() - t
        sum_elapsed += elapsed

        if myparams.isVerbose:
            print(
                "Iter {:4d} / {:d} | Rel_err= {:4.4e} | norm(dres) = {:4.1e} | "
                "norm(Q-Qtilde)/norm(Q) = {:4.2e} | t_cpu = {:2.2f}s | "
                "ranks_frame = ".format(
                    it,
                    myparams.maxit,
                    rel_err,
                    mu * norm_dres / norm_q,
                    rel_err_without_noise,
                    norm(reshape(E, -1)) / norm_q,
                    elapsed,
                ),
                *ranks
            )

        if it > 5 and np.abs(rel_err_list[-1] - rel_err_list[-4]) \
           < myparams.gtol * abs(rel_err_list[-1]):
            break

    qtilde = 0
    for p, (trafo_p, frame_p) in enumerate(zip(transforms, qtilde_frames)):
        qtilde += trafo_p.apply(frame_p.build_field())
        S = frame_p.modal_system["sigma"]
        frame_p.Nmodes = np.sum(S > 0)

    if myparams.isError:
        if myparams.isVerbose:
            print("CPU time in total: ", sum_elapsed)
        return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist, E)

    if myparams.isVerbose:
        print("CPU time in total: ", sum_elapsed)
    return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist)
    
### =================================================== ###
###                                                     ###
###             megaframe-based decompositions          ###
###                                                     ###    
### =================================================== ### 
    
def shifted_POD_J2_megaframe(
    snapshot_matrix,
    transforms,
    myparams,
    nmodes,
    nmodesstat=0
    ):
    """
    Megaframe implementation based on sPOD-J2
    """
    print("Method J2-megaframe")

    assert (
        np.ndim(snapshot_matrix) == 2
    ), "Please give enter a snapshotmatrix with every snapshot in one column"
    if myparams.use_rSVD:
        warn(
            "Warning: Using rSVD to accelarate decomposition procedure may lead "
            "to different results."
        )
    #########################    
    ## 1.Step: Initialize
    #########################
    nTransports = len(transforms)
    qtilde = np.zeros_like(snapshot_matrix)       # approximation of snapshot matrix 
    nRows = qtilde.shape[0]
    nCols = qtilde.shape[1]
    
    qmeg = np.zeros((nRows, nCols*nTransports))
    print(qmeg.shape)
    qstat = np.zeros_like(qtilde)

    it = 0
    rel_err = 1
    
    norm_q = np.linalg.norm(np.reshape(snapshot_matrix, -1))
    rel_err_list = [1.0]
    ranks_hist = []
    
    # calculate the residual
    res = snapshot_matrix - qtilde
    
    step = 1/nTransports
    sum_elapsed = 0
    while rel_err > myparams.eps and it < myparams.maxit:
        it += 1

        t = time.perf_counter()
        # update moving megaframe
        for k, trafo in enumerate(transforms):
            R_frame = trafo.reverse(res)
            qmeg[:, k*nCols:(k+1)*nCols] += R_frame*step
                        
        # truncate
        [umeg, smeg, vtmeg] = trunc_svd(qmeg, nmodes_max = nmodes, use_rSVD=myparams.use_rSVD)
        
        # (total variation smoothing)
        # AK: I'll probably smooth them block by block because there might be (physically legit) discontinuities between blocks
        if myparams.total_variation_iterations > 0:
            for k in range(1):
                VT = vtmeg[:, k*nCols:(k+1)*nCols]
                #VT = vtmeg
                VT = solve_TVL1(VT.T, clambda=myparams.tv_lambda, iter_n=myparams.total_variation_iterations).T
                vtmeg[:, k*nCols:(k+1)*nCols] = VT
                #vtmeg = VT
        
        rank = np.sum(smeg > 0)
        ranks = rank
        ranks_hist.append(rank)
        
        qmeg = np.dot((umeg*smeg), vtmeg)

        # update stationary frame
        if nmodesstat > 0:
            qstat += res        
            [ustat, sstat, vtstat] = trunc_svd(qstat, nmodes_max = nmodesstat, use_rSVD=myparams.use_rSVD)
            qstat = np.dot((ustat*sstat), vtstat)
        
            # update qtilde
            qtilde = qstat*step
        else:
            qtilde[...] = 0
        qhat = []
        for k, trafo in enumerate(transforms):
            qkhat = trafo.apply(qmeg[:, k*nCols:(k+1)*nCols])
            qhat.append(qkhat)
            qtilde += qkhat        #*step
        
        # calculate the residual
        res = snapshot_matrix - qtilde
        
        norm_res = np.linalg.norm(np.reshape(res, -1))
        rel_err = norm_res/norm_q
        rel_err_list.append(rel_err)
        
        elapsed = time.perf_counter() - t
        sum_elapsed += elapsed
        
        if myparams.isVerbose:
            print(
                "Iter %d / %d | Rel_err= %4.4e | rank_meg = %d | rank_stat = %d"%(
                    it,
                    myparams.maxit,
                    rel_err,
                    nmodes,
                    nmodesstat
                ),
            )
    
    # re-formate as a frame
    qtilde_frames = [Frame(transfo, field=qhat[k], Nmodes=nmodes) for k, transfo in enumerate(transforms)]
    # AK:   amplitudes get saved without smoothing for some reason?
    #       replace them with updated amplitudes with smoothing
    for k, frame in enumerate(qtilde_frames):
        frame.modal_system["U"] = umeg
        frame.modal_system["sigma"] = smeg
        frame.modal_system["VT"] = vtmeg[:, k*nCols:(k+1)*nCols]
    #qtilde_frames = [Frame(transfo, field=transfo.apply(qmeg[:, k*nCols:(k+1)*nCols]), Nmodes=nmodes) for k, transfo in enumerate(transforms)]        
         
    print("CPU time in total: ", sum_elapsed)
     
    if nmodesstat == 0:        
        return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist)
    else:
        # reformate qstat as frame
        transfostat = Transform([nRows, 1, 1, nCols], [1], transfo_type="identity")     # transform is identity
        stat_frame = Frame(transfostat, field=qstat, Nmodes=nmodesstat)
        stat_frame.modal_system["U"] = ustat
        stat_frame.modal_system["sigma"] = sstat
        stat_frame.modal_system["VT"] = vtstat
        
        return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist, stat_frame=stat_frame)
        
    #return qtilde, [umeg, smeg, vtmeg], [qstat, sstat, vtstat]
    
def shifted_POD_FB_megaframe(
    snapshot_matrix,
    transforms,
    myparams,
    nmodes_max=None,
    method = "JFB",
    nmodesstat = 0
):
    """
    This function implements the Forward-Backward method (FB).

    :param snapshot_matrix: Snapshot matrix with with dimensions :math:`M \times N`,
                            :math:`N` is the number of snapshots (i.e. time stamps)
                            and :math:`M` is the number of number of spatial samples
                            (i.e. the ODE dimension).
    :type snapshot_matrix: :class:`numpy.ndarray` (2-dimensional)

    :param transforms: List of transformations associated with the co-moving fields.
    :type transforms: list[Transform]

    :param nmodes: Number of modes to use in each frame
    :type nmodes: integer

    :param myparams: Parameters for the FB algorithm
    :type myparams: class:`sPOD_Param`

    :param method: Choice of the version of FB. Options are "BFB" (Block-coordinate
                   Forward Backward) and "JFB" (Joint Forward Backward).

    :return:
    :rtype: :class:`ReturnValue`
    """

    assert (
        np.ndim(snapshot_matrix) == 2
    ), "Please give enter a snapshotmatrix with every snapshot in one column"
    if myparams.use_rSVD:
        warn(
            "Warning: Using rSVD to accelarate decomposition procedure may lead "
            "to different results."
        )
    print('lambda_s = %f'%myparams.lambda_s)
    print('lambda_e = %f'%myparams.lambda_E)
    print(myparams.isError)

    ###################################
    #        1. Initialization        #
    ###################################
    nTransports = len(transforms)

    q = snapshot_matrix
    qtilde = np.zeros_like(q)
    
    nRows, nCols = np.shape(q)
    qmeg = np.zeros((nRows, nCols*nTransports))
    if myparams.stat_frame:
        qstat = np.zeros_like(q)
    if myparams.isError:
        E = np.zeros_like(snapshot_matrix)
    # make a list of the number of maximal ranks in each frame
    if not np.all(nmodes_max):  # check if array is None, if so set nmodes_max onto N
        nmodes_max = np.min(np.shape(snapshot_matrix)) # use the smallest dimension beacuse after this singular values will be 0

    norm_q = norm(reshape(q, -1))

    ###########################
    nmodes = 10
    # Error of the truncated SVD
    r_ = np.sum(nmodes)
    (u, s, vt) = trunc_svd(q, nmodes_max=None, use_rSVD=myparams.use_rSVD)
    err_svd = np.linalg.norm(q - np.dot(u * s, vt), ord="fro") / norm_q
    print(
        "Relative error using a truncated SVD with {:d} modes:{:4.4e}".format(
            r_, err_svd
        )
    )
    ###########################

    it = 0
    objective_0 = 0.5 * norm(q, ord="fro") ** 2
    objective_list = [objective_0]

    rel_err_list = [1.0]
    rel_decrease = 1
    rel_decrease_list = [1]
    ranks_hist = []
    sum_elapsed = 0
    res = q
    while rel_err > myparams.eps and it < myparams.maxit:
        it += 1
        ###################################
        #      2. Gradient step           #
        ###################################
        t = time.perf_counter()
        for k, trafo in enumerate(transforms):
            res_shifted = trafo.reverse(res)
            qkhat = qmeg[:, k*nCols:(k+1)*nCols]
            qmeg[:, k*nCols:(k+1)*nCols] += res_shifted / nTransports
            
        ###################################
        #      3. Proximal step           #
        ###################################
        [umeg, smeg, vtmeg] = trunc_SVT(qmeg, myparams.lambda_s, nmodes_max, myparams.use_rSVD)
        
        if myparams.stat_frame:
            qstat += res
            [ustat, sstat, vtstat] = trunc_SVT(qstat, myparams.lambda_stat, nmodes_max, myparams.use_rSVD)
            rank_stat = np.sum(sstat > 0)

        
        # (total variation smoothing)
        if myparams.total_variation_iterations > 0:
            for k in range(nTransports):
                VT = vtmeg[:, k*nCols:(k+1)*nCols]
                VT = solve_TVL1(VT.T, clambda=myparams.tv_lambda, iter_n=myparams.total_variation_iterations).T
                vtmeg[:, k*nCols:(k+1)*nCols] = VT
            if myparams.stat_frame:
                vtstat = solve_TVL1(vtstat.T, clambda=myparams.tv_lambda, iter_n=myparams.total_variation_iterations).T
                
        rank_meg = np.sum(smeg > 0)
        qmeg = np.dot(umeg*smeg, vtmeg)      # truncate
        ranks_hist.append(rank_meg)
                 

        ###################################
        #      4. Error matrix            #
        ###################################
        if myparams.isError:
            E = shrink(E + res / nTransports, myparams.lambda_E / nTransports)
        
        ###################################
        #      5. Update qtilde           #
        ###################################
        qtilde = np.zeros_like(q)
        
        if myparams.stat_frame:
            qstat = np.dot((ustat*sstat), vtstat)
            qtilde = qstat/nTransports
        
        qhat = []
        for k, trafo in enumerate(transforms):
            qkhat = trafo.apply(qmeg[:, k*nCols:(k+1)*nCols])
            qhat.append(qkhat)
            qtilde += qkhat
        
        ###################################
        #      6. Update residual         #
        ###################################
        if myparams.isError:
            res = q - qtilde - E
        else:
            res = q - qtilde
        norm_res = norm(reshape(res, -1))
        rel_err = norm_res / norm_q
        rel_err_list.append(rel_err)
        
        ###################################
        #   calculate objective function  #
        ###################################        
        if myparams.isError:
            objective = (0.5*norm(res, ord='fro')**2
                + myparams.lambda_s * norm(qmeg, ord='nuc') 
                + myparams.lambda_E * norm(E, ord=1))
        else:
            objective = (0.5*norm(res, ord='fro')**2
                + myparams.lambda_s * norm(qmeg, ord='nuc'))            
        objective_list.append(objective)
        rel_decrease = np.abs((objective_list[-1] - objective_list[-2])) / np.abs(
            objective_list[-1])
        rel_decrease_list.append(rel_decrease)
            
        elapsed = time.perf_counter() - t
        sum_elapsed += elapsed
        if myparams.isVerbose:
            if myparams.stat_frame:
                print(
                "Iter {:4d} / {:d} | Rel_err= {:4.4e} | t_cpu = {:2.2f}s | "
                "rank_meg = {:d} | rank_stat = {:d}".format(it, myparams.maxit, rel_err, elapsed, rank_meg, rank_stat),
                #rank_meg
            )
            else:    
                print(
                    "Iter {:4d} / {:d} | Rel_err= {:4.4e} | t_cpu = {:2.2f}s | "
                    "rank_meg = ".format(it, myparams.maxit, rel_err, elapsed),
                    rank_meg
                )
        
        if it > 5 and np.abs(rel_err_list[-1] - rel_err_list[-4]) \
           < myparams.gtol * abs(rel_err_list[-1]):
            break

        
    # reformulate as frames for output consistent with classical sPOD
    ranks = ranks_hist[-1]
    qtilde_frames = [Frame(transfo, field=qhat[k], Nmodes=ranks) for k, transfo in enumerate(transforms)]
    for k, frame in enumerate(qtilde_frames):
        frame.modal_system["U"] = umeg
        frame.modal_system["sigma"] = smeg
        frame.modal_system["VT"] = vtmeg[:, k*nCols:(k+1)*nCols]

    if myparams.isError:
        print("CPU time in total: ", sum_elapsed)
        return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist, E)
    elif myparams.stat_frame == False:
        print("CPU time in total: ", sum_elapsed)
        return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist)
        
    transfostat = Transform([nRows, 1, 1, nCols], [1], transfo_type="identity")     # transform is identity
    stat_frame = Frame(transfostat, field=qstat, Nmodes=rank_stat)
    stat_frame.modal_system["U"] = ustat
    stat_frame.modal_system["sigma"] = sstat
    stat_frame.modal_system["VT"] = vtstat
    print("CPU time in total: ", sum_elapsed)
    return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist, stat_frame=stat_frame)



def shifted_POD_ALM_megaframe(
    snapshot_matrix,
    transforms,
    myparams,
    nmodes_max=None,
    mu=None):
    """
    This function implements the Augmented Lagangian method (ALM).

    :param snapshot_matrix: Snapshot matrix with with dimensions :math:`M \times N`,
                            :math:`N` is the number of snapshots (i.e. time stamps)
                            and :math:`M` is the number of number of spatial samples
                            (i.e. the ODE dimension).
    :type snapshot_matrix: :class:`numpy.ndarray` (2-dimensional)

    :param transforms: List of transformations associated with the co-moving fields.
    :type transforms: List[Transform]

    :param myparams: Parameters for ALM algorithm
    :type myparams: class:`sPOD_Param`

    :param nmodes_max: Maximal number of modes allowed in each frame, default is
                       the number of snapshots :math:`N`.
                       Note: it is good to give a number large enough in order to
                             get the error down but smaller than :math:`N`.
                             This will increase the performance of the algorith.

    :param mu: Parameter of the augmented Lagrangian (i.e. weight of the
               quadratic term).
    :type mu: float

    :return:
    :rtype: :class:`ReturnValue`
    """
    """
    Megaframe implementation based on sPOD-ALM
    """
    assert (
        np.ndim(snapshot_matrix) == 2
    ), "Please give enter a snapshotmatrix with every snapshot in one column"
    if myparams.use_rSVD:
        warn(
            "Warning: Using rSVD to accelarate decomposition procedure may lead "
            "to different results."
        )
    ###################################
    #        1. Initialization        #
    ###################################
    qtilde = np.zeros_like(snapshot_matrix)
    nTransports = len(transforms)

    # make a list of the number of maximal ranks in each frame
    if not np.all(nmodes_max):  # check if array is None, if so set nmodes_max onto N
        nmodes_max = np.max(np.shape(snapshot_matrix))
    nmodes = nmodes_max
    #qtilde_frames = [
    #    Frame(transfo, qtilde, Nmodes=nmodes[k]) for k, transfo in enumerate(transforms)
    #]

    q = snapshot_matrix.copy()
    Y = np.zeros_like(snapshot_matrix)
    E = np.zeros_like(snapshot_matrix)
    if myparams.stat_frame:
        qstat = np.zeros_like(snapshot_matrix)
    norm_q = norm(reshape(q, -1))
    it = 0
    nRows, nCols = np.shape(q)
    if mu is None:
        mu = nCols * nRows / (4 * np.sum(np.abs(q)))
    if myparams.lambda_E is None:
        myparams.lambda_E = 1 / np.sqrt(np.maximum(nRows, nCols))
    if myparams.lambda_s is None:
        myparams.lambda_s = 1 / np.sqrt(np.maximum(nRows, nCols))   
    
    print('lambda_s = %f'%myparams.lambda_s)
    print('lambda_e = %f'%myparams.lambda_E)
    print('mu = %f'%mu)

    mu_inv = 1 / mu
    rel_err = 1
    res_old = 0
    rel_err_list = [1.0]
    ranks_hist = []
    sum_elapsed = 0
    
    res = q
    
    qtilde = np.zeros_like(q)
    qmeg = np.zeros((nRows, nCols*nTransports))     # megaframe
    qstat = 0

    while rel_err > myparams.eps and it < myparams.maxit:
        it += 1  # counts the number of iterations in the loop
        res = q - qtilde + mu_inv*Y

        ###################################
        #      3. Update the frames       #
        ###################################
        t = time.perf_counter()

        #qstat = np.zeros_like(q)            # stationary frame

        for k, trafo_k in enumerate(transforms):
            rk = trafo_k.reverse(res)
            qmeg[:, k*nCols : (k+1)*nCols] += rk / nTransports
        
        if myparams.stat_frame:
            qstat += res / nTransports
            [ustat, sstat, vtstat] = trunc_SVT(qstat, mu_inv * myparams.lambda_stat, nmodes, myparams.use_rSVD)
            rank_stat = np.sum(sstat>0)
            
        # proximal step (6):
        [umeg, smeg, vtmeg] = trunc_SVT(qmeg, mu_inv * myparams.lambda_s, nmodes, myparams.use_rSVD)
        rank_meg = np.sum(smeg > 0)
        
        # (total variation smoothing)
        if (myparams.total_variation_iterations > 0) and (rank_meg > 0):
            for k in range(nTransports):
                VT = vtmeg[:, k*nCols:(k+1)*nCols]
                VT = solve_TVL1(VT.T, clambda=myparams.tv_lambda, iter_n=myparams.total_variation_iterations).T
                vtmeg[:, k*nCols:(k+1)*nCols] = VT
            if ranks_stat > 0:
                vtstat = solve_TVL1(vtstat.T, clambda=myparams.tv_lambda, iter_n=myparams.total_variation_iterations).T
        
        qmeg = np.dot(umeg*smeg, vtmeg)      # truncate
        ranks_hist.append(rank_meg)

        # update reconstruction
        qtilde = np.zeros_like(q)
        qhat = []
        for k, trafo in enumerate(transforms):
            qkhat = trafo.apply(qmeg[:, k*nCols : (k+1)*nCols])
            qhat.append(qkhat) 
            qtilde += qkhat 
        
        if myparams.stat_frame:
            qstat = np.dot((ustat*sstat), vtstat)
            qtilde += qstat #/ nTransports
        
        ###################################
        #      4. Update noise term       #
        ###################################
        if myparams.isError:
            E = shrink(q - qtilde + mu_inv * Y, myparams.lambda_E * mu_inv)
    
        ###################################
        #      5. Update multipliers      #
        ###################################
        if myparams.isError:
            res = q - qtilde - E
        else:
            res = q - qtilde
        Y = Y + mu * res

        ###################################
        #          6. Update mu           #
        ###################################
        dres = norm(res, ord="fro") - res_old
        res_old = norm(res, ord="fro")
        norm_dres = np.abs(dres)

        norm_res = norm(reshape(res, -1))
        rel_err = norm_res / norm_q
        rel_err_list.append(rel_err)
        elapsed = time.perf_counter() - t
        sum_elapsed += elapsed

        if myparams.isVerbose:
            if myparams.stat_frame:
                print(
                "Iter {:4d} / {:d} | Rel_err= {:4.4e} | norm(dres) = {:4.1e} | "
                "rank_meg = {:d} | rank_stat = {:d}".format(it, 
                myparams.maxit, rel_err, mu * norm_dres / norm_q, rank_meg, rank_stat),)
            else:
                print(
                    "Iter {:4d} / {:d} | Rel_err= {:4.4e} | norm(dres) = {:4.1e} | "
                    "rank_meg = {:d} ".format(it, myparams.maxit, rel_err, mu * norm_dres / norm_q, rank_meg),)

        if it > 5 and np.abs(rel_err_list[-1] - rel_err_list[-4]) < myparams.gtol * abs(
            rel_err_list[-1]
        ):
            break

    ranks = ranks_hist[-1]
    # re-formate as a frame
    qtilde_frames = [Frame(transfo, field=qhat[k], Nmodes=nmodes) for k, transfo in enumerate(transforms)]
    for k, frame in enumerate(qtilde_frames):
        frame.modal_system["U"] = umeg
        frame.modal_system["sigma"] = smeg
        frame.modal_system["VT"] = vtmeg[:, k*nCols:(k+1)*nCols]
    
    if myparams.isError:
        print("CPU time in total: ", sum_elapsed)
        return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist, E)
        
    elif myparams.stat_frame == False:
        print("CPU time in total: ", sum_elapsed)
        return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist)
        
    transfostat = Transform([nRows, 1, 1, nCols], [1], transfo_type="identity")     # transform is identity
    stat_frame = Frame(transfostat, field=qstat, Nmodes=rank_stat)
    stat_frame.modal_system["U"] = ustat
    stat_frame.modal_system["sigma"] = sstat
    stat_frame.modal_system["VT"] = vtstat
    print("CPU time in total: ", sum_elapsed)
    return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist, stat_frame=stat_frame)


def force_constraint(qframes, transforms, q, Niter=1, alphas=None):
    """This function enforces the constraint 
    
    .. math::

        Q = sum_k T^k Q^k
    """
    norm_q = norm(reshape(q, -1))
    qtilde = np.zeros_like(q)
    if alphas == None:
        nTransports = len(transforms)
        alphas = [1 / nTransports] * nTransports

    for iter in range(Niter):
        qtilde = 0
        for k, (trafo, q_frame) in enumerate(zip(transforms, qframes)):
            qtilde += trafo.apply(q_frame.build_field())
            q_frame.Nmodes = -1

        res = q - qtilde
        qtilde = 0
        for k, (transfo, q_frame) in enumerate(zip(transforms, qframes)):
            res_shifted = transfo.reverse(res)
            q_frame_field = q_frame.build_field()
            q_frame.set_orthonormal_system(
                q_frame_field + res_shifted * alphas[k], use_rSVD=False
            )
            qtilde += transfo.apply(q_frame.build_field())

    res = q - qtilde
    norm_res = norm(reshape(res, -1))
    rel_err = norm_res / norm_q

    print("rel_err= {:4.4e}".format(rel_err))

    return ReturnValue(qframes, qtilde)


def give_interpolation_error(snapshot_data, transfo):
    """
    This function computes the interpolation error of the non-linear
    representation of the data.
    The error is "the best you can get" with the shifted POD.
    Therefore, it is the smallest possible error the shifted POD decomposition
    allows and can be used as the stopping criteria inside the sPOD algorithm.
    The relative error is computing using the Frobenius norm

    .. math::

        err = \| Q - T^(-1)[T[Q]] \|_F^2 / \| Q \|_F^2

    :param snapshot_data: Snapshot matrix.
    :type snapshot_data: :class:`numpy.ndarray` (2-dimensional)

    :param transfo:
    :type transfo: :class:`Transform`

    :return: Interpolation error.
    :rtype: float
    """
    from numpy import reshape

    Q = reshape(snapshot_data, [-1, snapshot_data.shape[-1]])
    rel_err = norm(Q - transfo.apply(transfo.reverse(Q)), ord="fro") / norm(
        Q, ord="fro"
    )
    return rel_err / 2


def save_frames(fname, frames, error_matrix=None):
    fname_base, old_ext = os.path.splitext(fname)
    ext = ".pkl"
    for k, frame in enumerate(frames):
        fname_frame = fname_base + "_%.2d" % k + ext
        print("frame %2d saved to: " % k, fname_frame)
        frame.save(fname_frame)
    if error_matrix is not None:
        fname_error_matrix = fname_base + "_error_mat.npy"
        np.save(fname_error_matrix, error_matrix)


def load_frames(fname, nTransports, load_ErrMat=False):
    fname_base, old_ext = os.path.splitext(fname)
    ext = ".pkl"

    # Load frames
    frame_list = []
    for k in range(nTransports):
        fname_frame = fname_base + "_%.2d" % k + ext
        print("frame %2d loaded: " % k, fname_frame)
        newframe = Frame(fname=fname_frame)
        frame_list.append(newframe)

    # Load sparse error matrix
    fname_error_matrix = fname_base + "_error_mat.npy"
    if load_ErrMat and os.path.isfile(fname_error_matrix):
        E = np.load(fname_error_matrix)
        return frame_list, E
    else:
        return frame_list


# ============================================================================ #


# ============================================================================ #
#                           CLASS of sPOD PARAMETERS                           #
# ============================================================================ #
@dataclass
class sPOD_Param:
    """
    Structure that stores the parameters for the sPOD algorithms.

    Attributes:
        gtol (float): Global tolerance for the stopping criterion of the algorithms.
        eps (float): Global tolerance for the stopping criterion of the algorithms.
        maxit (int): Maximum number of iterations.
        isVerbose (bool): Should the algorithm print information while running?
        isError (bool): Should the algorithm use the error term?
        use_rSVD (bool): Set to True in order to use randomized version of the SVD.
        lambda_s (float): Regularization parameter for the nuclear norm of the
                          co-moving frames.
        lambda_E (float): Regularization parameter for the l1-norm of the error
                          term.
        total_variation_itertations (int): Number of iterations of the iterative
                                           algorithm that computes the proximal
                                           operator of the total variation (TV).
        tv_lambda (float):  Parameter clambda for the total variation
        stat_frame (bool): Should megaframe use stationary frame?
        lambda_stat (float): Regularization parameter for the nuclear norm of the
                             stationary frame
    """

    gtol: float = 1e-6
    eps: float = 1e-6
    maxit: int = 10000
    isVerbose: bool = True
    isError: bool = False
    use_rSVD: bool = False
    lambda_s: float = 1e-2
    lambda_E: float = 1e-2
    total_variation_iterations: int = -1
    tv_lambda: float = 1.0
    
    stat_frame: bool = False
    lambda_stat: float = 1e-2


# ============================================================================ #

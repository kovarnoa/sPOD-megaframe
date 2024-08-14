#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Mar 06 15:08:42 2024

@author: Philipp Krah, Beata Zorawski, Arthur Marmin, Anna Kovarnova

Comparison of performance of normal sPOD vs. megaframe.
"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
import sys
from generate_data import generate_data

sys.path.append("../lib")
import numpy as np
from numpy import mod, meshgrid, cos, sin, exp, pi
import matplotlib.pyplot as plt

#from zaloha_sPOD_algo import(
#from sPOD_algo_GS import(
from sPOD_algo import (
    shifted_POD,
    sPOD_Param,
    give_interpolation_error,
)
from transforms import Transform
from plot_utils import save_fig

# ============================================================================ #


# ============================================================================ #
#                              CONSTANT DEFINITION                             #
# ============================================================================ #
PIC_DIR = "../images/"
SAVE_FIG = True
PLOT_VT = True

#CASE = "crossing_waves"
#CASE = "three_cross_waves"        
#CASE = "sine_waves"
#CASE = "sine_waves_noise"
CASE = "multiple_ranks"
#CASE = "non_cross_mult_ranks"

Nx = 400        # number of grid points in x
Nt = Nx // 2    # number of time intervals
Niter = 100      # number of sPOD iterations

#VARIANT = "ALM"
#VARIANT = "JFB"
VARIANT = "J2"

# ============================================================================ #


# ============================================================================ #
#                                 Main Program                                 #
# ============================================================================ #
# Clean-up
plt.close("all")
# Data Generation
fields, shift_list, nmodes_exact, L, dx = generate_data(Nx, Nt, CASE)

############################################
# %% CALL THE SPOD algorithm
############################################
data_shape = [Nx, 1, 1, Nt]
transfos = [
    Transform(data_shape, [L], shifts=shift_list[0], dx=[dx], interp_order=5),
    Transform(data_shape, [L], shifts=shift_list[1], dx=[dx], interp_order=5),
]

if CASE == "three_cross_waves":
   transfos.append(Transform(data_shape, [L], shifts=shift_list[2], dx=[dx], interp_order=5))

interp_err = np.max([give_interpolation_error(fields, transfo) for transfo in transfos])
print("interpolation error: {:1.2e}".format(interp_err))
# %%
qmat = np.reshape(fields, [Nx, Nt])
mu0 = Nx * Nt / (4 * np.sum(np.abs(qmat))) * 0.1
#lambd0 = 20
lambd0 = 1
myparams = sPOD_Param()
myparams.maxit = Niter
param_alm = None
nmodes = None

############################################
# %% run normal variant
############################################
myparams.isError = False
if (VARIANT != "J2") and ((CASE == "sine_waves_noise") or (CASE == "sine_waves")):
    myparams.isError = True

myparams.total_variation_iterations = 80
myparams.tv_lambda = 1      # the smaller tv_lambda, the bigger smoothing of time amplitudes

METHOD = VARIANT
if METHOD == "ALM":
    param_alm = mu0  # adjust for case
    myparams.lambda_s = 1
    if CASE == "multiple_ranks":
        myparams.lambda_s = 4
        param_alm = mu0*0.08
    if CASE == "sine_waves":
        myparams.lambda_s = 0.35
        myparams.lambda_E = 0.0135
elif METHOD == "JFB":
    myparams.lambda_s = 0.2
    if CASE == "non_cross_mult_ranks":
        myparams.lambda_s = 0.7
    if CASE == "multiple_ranks":
        myparams.lambda_s = 1
    if CASE == "sine_waves":
        myparams.lambda_s = 0.35
        myparams.lambda_E = 0.0135
    if CASE == "sine_waves_noise":
        myparams.lambda_s = 1.6
        myparams.lambda_E = 0.07

elif METHOD == "J2":
    nmodes = nmodes_exact
    if CASE == "sine_waves_noise":
        print()
        print("J2 algorithms fail for cases with noise -- choose ALM or JFB")
        #input("If you wish to continue anyway, press Enter...")

ret = shifted_POD(qmat, transfos, myparams, METHOD, param_alm, nmodes=nmodes)

sPOD_frames, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
qf = [
    np.squeeze(np.reshape(transfo.apply(frame.build_field()), data_shape))
    for transfo, frame in zip(transfos, ret.frames)
]

#np.savetxt('./rel_err.dat', rel_err, fmt='%05.4e')
rank = ret.ranks


if PLOT_VT:
    for k, frame in enumerate(sPOD_frames):
        #print(sPOD_frames_meg[k].modal_system["sigma"][:rank_meg])
        VT = sPOD_frames[k].modal_system["VT"]
        VT = VT[:rank[k], :]
        np.savetxt(PIC_DIR + "vt_classic_%d_%s.dat"%(k, VARIANT), VT.T, fmt='%03.2e', delimiter='\t')
    
        for indM in range(rank[k]):   
            fig = plt.figure()
            plt.plot(VT[indM,:])
            plt.xlim(0, Nt)
            plt.savefig(PIC_DIR + "tvl_test/vt_classic_%d_%d_%s_%d.png"%(k, indM, VARIANT, myparams.total_variation_iterations))
            plt.close()
            
            # AK: only first ten modes, otherwise takes too long
            if indM > 10: 
                break

        
print()

############################################
# %% run megaframe variant
############################################
myparams.maxit = 100

METHOD = VARIANT + "_megaframe"
if METHOD == "J2_megaframe":
    nmodes = np.max(nmodes_exact)
if METHOD == "JFB_megaframe":
    myparams.lambda_s = 0.1
    if CASE == "multiple_ranks":
        myparams.lambda_s = 0.6
    if CASE == "non_cross_mult_ranks":
        myparams.lambda_s = 0.5
    if CASE == "sine_waves":
        myparams.lambda_s = 0.32        # with noise
        #myparams.lambda_s = 0.43        # without noise
        myparams.lambda_E = 0.018
    if CASE == "sine_waves_noise":
        myparams.lambda_s = 0.8
        myparams.lambda_E = 0.055
if METHOD == "ALM_megaframe":
    myparams.lambda_s = 2 # adjust for case
    param_alm = mu0*0.1
    if CASE == "multiple_ranks":
        myparams.lambda_s = 2.5
        param_alm = mu0*0.2
    if CASE == "sine_waves":
        param_alm = mu0*0.5
        myparams.lambda_s = 1.6
        myparams.lambda_E = 0.05
    if CASE == "sine_waves_noise":
        myparams.lambda_s = 2.6
        myparams.lambda_E = 0.18
        param_alm = mu0*0.3

ret_meg = shifted_POD(qmat, transfos, myparams, METHOD, param_alm, nmodes=nmodes)

sPOD_frames_meg, qtilde_meg, rel_err_meg = ret_meg.frames, ret_meg.data_approx, ret_meg.rel_err_hist
#np.savetxt('./rel_err_meg.dat', rel_err_meg, fmt='%05.4e')

rank_meg = ret_meg.ranks
qf_meg = [
    np.squeeze(np.reshape(transfo.apply(frame.build_field()), data_shape))
    for transfo, frame in zip(transfos, ret_meg.frames)
]

if PLOT_VT:
    VTmeg = np.zeros((rank_meg, Nt*len(shift_list)))
    for k, frame in enumerate(sPOD_frames_meg):
        VT = sPOD_frames_meg[k].modal_system["VT"]
        VT = VT[:rank_meg, :]
        VTmeg[:, k*Nt:(k+1)*Nt] = VT
        np.savetxt(PIC_DIR + "vt_%d_%s.dat"%(k, VARIANT), VT.T, fmt='%03.2e', delimiter='\t')
    
    for indM in range(min(rank_meg, 10)):   
        fig = plt.figure()
        plt.plot(VTmeg[indM,:])
        plt.xlim(0, Nt*2)
        plt.savefig(PIC_DIR + "vt_%d_%s_%d.png"%(indM, VARIANT, myparams.total_variation_iterations))
        plt.close()


############################################
# %% visualize your results: sPOD frames
############################################
# first we plot the resulting field
# cases with three frames or an error matrix will need a larger figure
if (CASE == "three_cross_waves") or (myparams.isError == True):
    gridspec = {"width_ratios": [1, 1, 1, 1, 1]}
    fig, ax = plt.subplots(2, 5, figsize=(14, 7), gridspec_kw=gridspec, num=101, layout='constrained')
else:
    gridspec = {"width_ratios": [1, 1, 1, 1]}
    fig, ax = plt.subplots(2, 4, figsize=(12, 7), gridspec_kw=gridspec, num=101, layout='constrained')    
mycmap = "viridis"
vmin = np.min(qmat) * 0.6
vmax = np.max(qmat) * 0.6

############## megaframe results #######################
ax[0,0].pcolormesh(qmat, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[0,0].set_title(r"$\mathbf{Q}$")
# ax[0].axis("image")
ax[0,0].axis("off")

ax[0,1].pcolormesh(qtilde_meg, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[0,1].set_title(r"$\tilde{\mathbf{Q}}$ -- megaframe rank " + str(ret_meg.ranks))
# ax[0].axis("image")
ax[0,1].axis("off")
# the result is a list of the decomposed field.
# each element of the list contains a frame of the decomposition.
# If you want to plot the k-th frame use:
# 1. frame
plot_shifted = False
k_frame = 0
if plot_shifted:
    ax[0,2].pcolormesh(qf_meg[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[0,2].set_title(r"$T^" + str(k_frame + 1) + "\hat{\mathbf{Q}}^" + str(k_frame + 1) + "$")
else:
    ax[0,2].pcolormesh(sPOD_frames_meg[k_frame].build_field(), vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[0,2].set_title(r"$\hat{\mathbf{Q}}^" + str(k_frame + 1) + "$")
ax[0,2].axis("off")
# ax[1].axis("image")
# 2. frame
k_frame = 1
if plot_shifted:
    im2 = ax[0,3].pcolormesh(qf_meg[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[0,3].set_title(r"$T^" + str(k_frame + 1) + "\hat{\mathbf{Q}}^" + str(k_frame + 1) + "$")
else:
    im2 = ax[0,3].pcolormesh(sPOD_frames_meg[k_frame].build_field(), vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[0,3].set_title(r"$\hat{\mathbf{Q}}^" + str(k_frame + 1) + "$")
ax[0,3].axis("off")
# ax[2].axis("image")

if CASE == "three_cross_waves":
    # 3rd frame
    k_frame = 2
    if plot_shifted:
        im2 = ax[0,4].pcolormesh(qf_meg[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
        ax[0,4].set_title(r"$T^" + str(k_frame + 1) + "\hat{\mathbf{Q}}^" + str(k_frame + 1) + "$")
    else:
        im2 = ax[0,4].pcolormesh(sPOD_frames_meg[k_frame].build_field(), vmin=vmin, vmax=vmax, cmap=mycmap)
        ax[0,4].set_title(r"$\hat{\mathbf{Q}}^" + str(k_frame + 1) + "$")
    ax[0,4].axis("off")
elif myparams.isError == True:
    # extra plotting error matrix
    im2 = ax[0,4].pcolormesh(ret_meg.error_matrix, vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[0,4].set_title(r"$\mathbf{E}$")
    ax[0,4].axis("off")


############## classic sPOD results #######################
# ax[0].axis("image")
ax[1,0].axis("off")

ax[1,1].pcolormesh(qtilde, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[1,1].set_title(r"$\tilde{\mathbf{Q}}$ -- classical -- over. rank " + str(sum(ret.ranks)))
# ax[0].axis("image")
ax[1,1].axis("off")
# the result is a list of the decomposed field.
# each element of the list contains a frame of the decomposition.
# If you want to plot the k-th frame use:
# 1. frame
plot_shifted = False
k_frame = 0
if plot_shifted:
    ax[1,2].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[1,2].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    ax[1,2].pcolormesh(sPOD_frames[k_frame].build_field(), vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[1,2].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[1,2].axis("off")
# ax[1].axis("image")
# 2. frame
k_frame = 1
if plot_shifted:
    im2 = ax[1,3].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[1,3].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    im2 = ax[1,3].pcolormesh(sPOD_frames[k_frame].build_field(), vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[1,3].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[1,3].axis("off")
# ax[2].axis("image")

if CASE == "three_cross_waves":
    k_frame = 2
    if plot_shifted:
        im2 = ax[1,4].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
        ax[1,4].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
    else:
        im2 = ax[1,4].pcolormesh(sPOD_frames[k_frame].build_field(), vmin=vmin, vmax=vmax, cmap=mycmap)
        ax[1,4].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
    ax[1,4].axis("off")
elif (myparams.isError == True):
    # extra plotting error matrix
    im2 = ax[1,4].pcolormesh(ret.error_matrix, vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[1,4].set_title(r"$\mathbf{E}$")
    ax[1,4].axis("off")

for row in range(2):
    for axes in ax[row, :]:
        axes.set_aspect(0.6)

if (CASE == "three_cross_waves") or (myparams.isError == True):
    plt.colorbar(im2, ax=ax[0, 4], location='right', shrink=0.8)
else:
    plt.colorbar(im2, ax=ax[0, 3], location='right', shrink=0.8)
#plt.tight_layout()

if SAVE_FIG:
    plt.savefig(PIC_DIR + "01_pictures_%s_%s_%d.png"%(CASE, VARIANT, myparams.total_variation_iterations))
#plt.show()

#######################################################
###                 Plot convergence                ###
#######################################################

plt.close(11)
xlims = [-1, Niter]
ylims = [np.min(ret.ranks_hist[:])-1, max(np.max(ret.ranks_hist[:]), np.max(ret_meg.ranks_hist[:]))+1]
ylims2 = [min(min(rel_err), min(rel_err_meg)), max(max(rel_err), max(rel_err_meg))]

fig, ax = plt.subplots(1, 2, figsize=(12, 4), num=11)

# classic #############################
ax[0].set_title("Classical sPOD formulation -- %s"%VARIANT)

ax[0].plot(ret.ranks_hist[0], "+", color='tab:blue', label="$\mathrm{rank}(\mathbf{Q}^1)$")
ax[0].plot(ret.ranks_hist[1], "x", color='tab:cyan', label="$\mathrm{rank}(\mathbf{Q}^2)$")
ax[0].plot(xlims, [nmodes_exact[0], nmodes_exact[0]], "k--", label="exact rank $r_1=%d$" % nmodes_exact[0])
ax[0].plot(xlims, [nmodes_exact[1], nmodes_exact[1]], "k-", label="exact rank $r_2=%d$" % nmodes_exact[1])
ax[0].set_xlim(xlims)
ax[0].set_ylim(ylims)
ax[0].set_xlabel("iterations")
ax[0].set_ylabel("rank $r_k$", color='tab:blue')
ax[0].tick_params(axis='y', labelcolor='tab:blue')
ax[0].legend()

ax0 = ax[0].twinx()
ax0.plot(rel_err, color='tab:orange')
ax0.set_ylabel('Rel. error', color='tab:orange')
ax0.set_ylim(ylims2)
ax0.set_yscale('log')
ax0.tick_params(axis='y', labelcolor='tab:orange')

# megaframe ##########################
ax[1].set_title("Megaframe sPOD formulation -- %s"%VARIANT)

ax[1].plot(ret_meg.ranks_hist, "+", color='tab:blue', label="$\mathrm{rank}(\mathbf{\hat{Q}})$")
ax[1].plot(xlims, [nmodes_exact[0], nmodes_exact[0]], "k--", label="exact rank $r=%d$" %max(nmodes_exact))
ax[1].set_xlim(xlims)
ax[1].set_ylim(ylims)
ax[1].set_xlabel("iterations")
ax[1].set_ylabel("rank $r_{meg}$", color='tab:blue')
ax[1].tick_params(axis='y', labelcolor='tab:blue')
ax[1].legend()

ax1 = ax[1].twinx()
ax1.plot(rel_err_meg, color='tab:orange')
ax1.set_ylabel('Rel. error', color='tab:orange')
ax1.set_yscale('log')
ax1.set_ylim(ylims2)
ax1.tick_params(axis='y', labelcolor='tab:orange')

plt.tight_layout()


if SAVE_FIG:
    plt.savefig(PIC_DIR + "01_convergence_%s_%s_%d.png"%(CASE, VARIANT, myparams.total_variation_iterations))
plt.show()


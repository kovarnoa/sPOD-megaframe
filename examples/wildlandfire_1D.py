# -*- coding: utf-8 -*-
"""
@author: Philipp Krah, Beata Zorawski, Arthur Marmin
"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
import sys

sys.path.append("../lib")
import numpy as np
from numpy import meshgrid
import matplotlib.pyplot as plt
from sPOD_algo import (
    shifted_POD,
    sPOD_Param,
    give_interpolation_error,
)
from transforms import Transform
from plot_utils import save_fig

# ============================================================================ #


def generate_wildlandfire_data(grid, time, snapshot, shifts):
    x = np.load(grid, allow_pickle=True)[0]
    t = np.load(time)
    L = x[-1]
    Nx = len(x)
    Nt = len(t)
    q = np.load(snapshot)
    Q = q[:Nx, :]  # Temperature
    # Q = q[Nx:, :] # Supply mass
    shift_list = np.load(shifts)
    for k in range(shift_list.shape[0]):
        start = shift_list[k, 0]
        shift_list[k, :] -= start
   
    [X, T] = meshgrid(x, t)
    X = X.T
    T = T.T
    dx = x[1] - x[0]
    nmodes = [1, 1, 1]
    mirroring = [False, False, True]

    return Q, shift_list, L, dx, Nx, Nt, nmodes


SAVE_FIG = True
PIC_DIR = "../images/"
PLOT_VT = False
Niter = 100

fields, shift_list, L, dx, Nx, Nt, nmodes = generate_wildlandfire_data(
    f"../examples/Wildlandfire_1d/1D_Grid.npy",
    f"../examples/Wildlandfire_1d/Time.npy",
    f"../examples/Wildlandfire_1d/SnapShotMatrix558.49.npy",
    f"../examples/Wildlandfire_1d/Shifts558.49.npy",
)

data_shape = [Nx, 1, 1, Nt]
print("Data shape = " + str(data_shape))
transfos = [
    Transform(data_shape, [L], transfo_type="shift", shifts=shift_list[0], dx=[dx], interp_order=5),
    Transform(data_shape, [L], transfo_type="shift", shifts=shift_list[1], dx=[dx], interp_order=5),
    Transform(data_shape, [L], transfo_type="shift", shifts=shift_list[2], dx=[dx], interp_order=5),
]

interp_err = np.max([give_interpolation_error(fields, trafo) for trafo in transfos])
print("interpolation error: %1.2e " % interp_err)

qmat = np.reshape(fields, [Nx, Nt])

VARIANT = "J2"              # AK: megaframe uses stationary frame, which is so far only implemented in J2 (:
#VARIANT = "ALM"
# VARIANT = "JFB"
# VARIANT = "BFB"
METHOD = VARIANT
lambda0 = 4000  # for Temperature
# lambda0 = 27  # for supply mass
myparams = sPOD_Param()
myparams.maxit = Niter
param_alm = None
nmodes_max = None
mu0 = Nx * Nt / (4 * np.sum(np.abs(qmat)))

if METHOD == "ALM":
    myparams.lambda_s = 1
    param_alm = mu0 * 0.01
elif METHOD == "BFB":
    myparams.lambda_s = lambda0
elif METHOD == "JFB":
    myparams.lambda_s = lambda0
elif METHOD == "J2":
    #nmodes_max = [3, 2, 3]      # results from JFB, but it's pretty arbitrary
    nmodes_max = [3, 1, 3]      # ALM

print()
print("Classical")
print()

ret = shifted_POD(qmat, transfos, myparams, METHOD, param_alm, nmodes=nmodes_max)

sPOD_frames, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
qf = [
    np.squeeze(np.reshape(trafo.apply(frame.build_field()), data_shape))
    for trafo, frame in zip(transfos, ret.frames)
]


###################### Megaframe #######################################
print()
print("Megaframe")
print()
transfos = [
    Transform(data_shape, [L], transfo_type="shiftMirror", shifts=shift_list[0], dx=[dx], is_mirrored=False, interp_order=5),
    # second is in stationary frame
    # Transform(data_shape, [L], transfo_type="shiftMirror", shifts=shift_list[1], dx=[dx], is_mirrored=False, interp_order=5),
    Transform(data_shape, [L], transfo_type="shiftMirror", shifts=shift_list[2], dx=[dx], is_mirrored=True, interp_order=5),
]
interp_err = np.max([give_interpolation_error(fields, trafo) for trafo in transfos])

nmodesstat = nmodes_max[1]
nmodesmeg = nmodes_max[0]
METHOD = VARIANT + "_megaframe"
if nmodesstat > 0:
    ret_meg, qf_stat = shifted_POD(qmat, transfos, myparams, METHOD, param_alm, nmodes=nmodesmeg, nmodesstat=nmodesstat)
    qstat = np.squeeze(np.reshape(qf_stat.build_field(), data_shape))
    rank_stat = sum(qf_stat.modal_system["sigma"] > 0)
    print(rank_stat)
else:
    ret_meg = shifted_POD(qmat, transfos, myparams, METHOD, param_alm, nmodes=nmodes_max, nmodesstat=nmodesstat)
    rank_stat = 0

sPOD_frames_meg, qtilde_meg, rel_err_meg = ret_meg.frames, ret_meg.data_approx, ret_meg.rel_err_hist
qf_meg = [
    np.squeeze(np.reshape(trafo.apply(frame.build_field()), data_shape))
    for trafo, frame in zip(transfos, ret_meg.frames)
]

rank_meg = ret_meg.ranks
#np.savetxt('./rel_err_meg', rel_err_meg, fmt='%05.4e')

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
            plt.xlim(0, Nt*len(transfos))
            plt.savefig(PIC_DIR + "vt_%d_%d_%s.png"%(k, indM, VARIANT))
            plt.close()
    
    if nmodesstat > 0:
        VTstat = qf_stat.modal_system["VT"]
        np.savetxt(PIC_DIR + "vt_stat_%s.dat"%(VARIANT), VT.T, fmt='%03.2e', delimiter='\t')
        for indM in range(min(rank_stat, 10)):   
            fig = plt.figure()
            plt.plot(VTmeg[indM,:])
            plt.xlim(0, Nt)
            plt.savefig(PIC_DIR + "vt_stat_%d_%s.png"%(indM, VARIANT))
            plt.close()


# %% 1. visualize your results: sPOD frames
##########################################
# first we plot the resulting field
gridspec = {"width_ratios": [1, 1, 1, 1, 1]}
fig, ax = plt.subplots(2, 5, figsize=(14, 7), gridspec_kw=gridspec, num=101, layout='constrained')
mycmap = "viridis"
vmin = np.min(qmat) * 0.6
vmax = np.max(qmat) * 0.6

ax[0, 0].pcolormesh(qmat, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[0, 0].set_title(r"$\mathbf{Q}$")
ax[0, 0].axis("off")

## Megaframe
ax[0, 1].pcolormesh(qtilde_meg, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[0, 1].set_title(r"$\tilde{\mathbf{Q}}$ -- megaframe rank " + str(rank_meg+rank_stat))
ax[0, 1].axis("off")

# the result is a list of the decomposed field.
# each element of the list contains a frame of the decomposition.
# If you want to plot the k-th frame use:
# 1. frame
plot_shifted = False
k_frame = 0
if plot_shifted:
    ax[0, 2].pcolormesh(qf_meg[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[0, 2].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{\hat{Q}}^" + str(k_frame + 1) + "$")
else:
    ax[0, 2].pcolormesh(sPOD_frames_meg[k_frame].build_field(), vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[0, 2].set_title(r"$\mathbf{\hat{Q}}^" + str(k_frame + 1) + "$")
ax[0, 2].axis("off")

im2 = ax[0, 3].pcolormesh(qstat, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[0, 3].set_title(r"$\mathbf{Q}^{\mathrm{stat}}$")
ax[0, 3].axis("off")

k_frame = 1
if plot_shifted:
    im2 = ax[0, 4].pcolormesh(qf_meg[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[0, 4].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{\hat{Q}}^" + str(k_frame + 1) + "$")
else:
    im2 = ax[0, 4].pcolormesh(sPOD_frames_meg[k_frame].build_field(), vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[0, 4].set_title(r"$\mathbf{\hat{Q}}^" + str(k_frame + 1) + "$")
ax[0, 4].axis("off")

### Classical sPOD
ax[1, 0].axis("off")

ax[1, 1].pcolormesh(qtilde_meg, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[1, 1].set_title(r"$\tilde{\mathbf{Q}}$ -- classical rank " + str(sum(nmodes_max)))
ax[1, 1].axis("off")

# the result is a list of the decomposed field.
# each element of the list contains a frame of the decomposition.
# If you want to plot the k-th frame use:
# 1. frame
plot_shifted = False
k_frame = 0
if plot_shifted:
    ax[1, 2].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[1, 2].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    ax[1, 2].pcolormesh(sPOD_frames[k_frame].build_field(), vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[1, 2].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[1, 2].axis("off")

k_frame = 1
if plot_shifted:
    ax[1, 3].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[1, 3].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    ax[1, 3].pcolormesh(sPOD_frames[k_frame].build_field(), vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[1, 3].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[1, 3].axis("off")

k_frame = 2
if plot_shifted:
    ax[1, 4].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[1, 4].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    ax[1, 4].pcolormesh(sPOD_frames[k_frame].build_field(), vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[1, 4].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[1, 4].axis("off")


for row in range(2):
    for axes in ax[row, :]:
        axes.set_aspect(0.25)
        
plt.colorbar(im2)

if SAVE_FIG:
    plt.savefig(PIC_DIR + "01_pictures_wildlandfire_%s.png"%(METHOD))
plt.show()


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
ax[0].plot(ret.ranks_hist[2], "x", color='tab:gray', label="$\mathrm{rank}(\mathbf{Q}^3)$")
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
ax[1].plot(rank_stat, "x", color="tab:cyan", label="rank( $\mathbf{Q}_{\mathrm{stat}})$")
ax[1].set_xlim(xlims)
ax[1].set_ylim(ylims)
ax[1].set_xlabel("iterations")
ax[1].set_ylabel("rank", color='tab:blue')
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
    plt.savefig(PIC_DIR + "01_convergence_wildlandfire_%s.png"%(VARIANT))
plt.show()


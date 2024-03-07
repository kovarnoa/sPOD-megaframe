#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Nov 30 09:11:12 2018

@author: philipp krah, jiahan wang
"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
import sys

sys.path.append("../lib")
import numpy as np
from numpy import exp, mod, meshgrid, cos, sin, exp, pi
import matplotlib.pyplot as plt
from sPOD_algo import (
    shifted_POD_J2,
    shifted_POD_FB,
    shifted_POD_ALM,
    sPOD_Param,
    give_interpolation_error,
)
from transforms import Transform
from plot_utils import save_fig

# ============================================================================ #

pic_dir = "../images/"


def generate_data(Nx, Nt, case, noise_percent=0.2):
    Tmax = 0.5  # total time
    L = 1  # total domain size
    sigma = 0.015  # standard diviation of the puls
    x = np.arange(0, Nx) / Nx * L
    t = np.arange(0, Nt) / Nt * Tmax
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    c = 1
    [X, T] = meshgrid(x, t)
    X = X.T
    T = T.T

    if case == "crossing_waves":
        nmodes = 1
        fun = lambda x, t: exp(-((mod((x - c * t), L) - 0.1) ** 2) / sigma**2) + exp(
            -((mod((x + c * t), L) - 0.9) ** 2) / sigma**2
        )

        # Define your field as a list of fields:
        # For example the first element in the list can be the density of
        # a flow quantity and the second element could be the velocity in 1D
        density = fun(X, T)
        velocity = fun(X, T)
        shifts1 = np.asarray(-c * t)
        shifts2 = np.asarray(c * t)
        Q = density  # , velocity]
        shift_list = [shifts1, shifts2]
    elif case == "sine_waves":
        delta = 0.0125
        # first frame
        q1 = np.zeros_like(X)
        shifts1 = -0.25 * cos(7 * pi * t)
        for r in np.arange(1, 5):
            x1 = 0.25 + 0.1 * r - shifts1
            q1 = q1 + sin(2 * pi * r * T / Tmax) * exp(-((X - x1) ** 2) / delta**2)
        # second frame
        c2 = dx / dt
        shifts2 = -c2 * t
        q2 = np.zeros_like(X)

        x2 = 0.2 - shifts2
        q2 = exp(-((X - x2) ** 2) / delta**2)

        Q = q1 + q2
        nmodes = [4, 1]
        shift_list = [shifts1, shifts2]

    elif case == "sine_waves_noise":
        delta = 0.0125
        # first frame
        q1 = np.zeros_like(X)
        shifts1 = -0.25 * cos(7 * pi * t)
        for r in np.arange(1, 5):
            x1 = 0.25 + 0.1 * r - shifts1
            q1 = q1 + sin(2 * pi * r * T / Tmax) * exp(-((X - x1) ** 2) / delta**2)
        # second frame
        c2 = dx / dt
        shifts2 = -c2 * t

        x2 = 0.2 - shifts2
        q2 = exp(-((X - x2) ** 2) / delta**2)
        # E = 0.1*np.random.random(np.shape(q2))
        Q = q1 + q2  # + E
        indices = np.random.choice(
            np.arange(Q.size), replace=False, size=int(Q.size * noise_percent)
        )
        Q = Q.flatten()
        Q[indices] = 1
        Q = np.reshape(Q, np.shape(q1))
        nmodes = [4, 1]
        shift_list = [shifts1, shifts2]

    elif case == "multiple_ranks":
        delta = 0.0125
        # first frame
        q1 = np.zeros_like(X)
        c2 = dx / dt
        shifts1 = c2 * t
        for r in np.arange(1, 5):
            # x1 = 0.25 + 0.1 * r - shifts1
            x1 = 0.5 + 0.1 * r - shifts1
            q1 = q1 + sin(2 * pi * r * T / Tmax) * exp(-((X - x1) ** 2) / delta**2)
        # second frame
        c2 = dx / dt
        shifts2 = -c2 * t
        q2 = np.zeros_like(X)
        for r in np.arange(1, 3):
            x2 = 0.2 + 0.1 * r - shifts2
            q2 = q2 + cos(2 * pi * r * T / Tmax) * exp(-((X - x2) ** 2) / delta**2)

        Q = q1 + q2
        nmodes = [4, 2]
        shift_list = [shifts1, shifts2]

    return Q, shift_list, nmodes, L, dx


# ============================================================================ #

##########################################
# %% Define parameters:
##########################################
plt.close("all")
case = "multiple_ranks"
# case = "sine_waves"
Nx = 400  # number of grid points in x
Nt = Nx // 2  # number of time intervals
Niter = 500  # number of sPOD iterations
# method = "ALM"
method = "BFB"
# method = "JFB"
# method = "J2"

fields, shift_list, nmodes, L, dx = generate_data(Nx, Nt, case)
#######################################
# %% CALL THE SPOD algorithm
######################################
data_shape = [Nx, 1, 1, Nt]
transfos = [
    Transform(data_shape, [L], shifts=shift_list[0], dx=[dx], interp_order=5),
    Transform(data_shape, [L], shifts=shift_list[1], dx=[dx], interp_order=5),
]

interp_err = np.max([give_interpolation_error(fields, transfo) for transfo in transfos])
print("interpolation error: {:1.2e}".format(interp_err))
# %%
qmat = np.reshape(fields, [Nx, Nt])
mu0 = Nx * Nt / (4 * np.sum(np.abs(qmat)))
lambd0 = 1 / np.sqrt(np.maximum(Nx, Nt))
myparams = sPOD_Param()
myparams.maxit = Niter
if method == "ALM":
    nmodes_param = np.max(nmodes) + 10
    ret = shifted_POD_ALM(
        qmat,
        transfos,
        myparams,
        nmodes_max=nmodes_param,
        use_rSVD=False,
        mu=mu0,  # adjust for case
    )
elif method == "BFB":
    myparams.lambda_s = 0.3  # adjust for case
    ret = shifted_POD_FB(qmat, transfos, nmodes, myparams, method="BFB")
elif method == "JFB":
    myparams.lambda_s = 0.4  # adjust for case
    ret = shifted_POD_FB(qmat, transfos, nmodes, myparams, method="JFB")
elif method == "J2":
    ret = shifted_POD_J2(qmat, transfos, nmodes, myparams, True)
sPOD_frames, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
qf = [
    np.squeeze(np.reshape(transfo.apply(frame.build_field()), data_shape))
    for transfo, frame in zip(transfos, ret.frames)
]
###########################################
# %% 1. visualize your results: sPOD frames
##########################################
# first we plot the resulting field
gridspec = {"width_ratios": [1, 1, 1, 1]}
fig, ax = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw=gridspec, num=101)
mycmap = "viridis"
vmin = np.min(qtilde) * 0.6
vmax = np.max(qtilde) * 0.6

ax[0].pcolormesh(qmat, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[0].set_title(r"$\mathbf{Q}$")
# ax[0].axis("image")
ax[0].axis("off")

ax[1].pcolormesh(qtilde, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[1].set_title(r"$\tilde{\mathbf{Q}}$")
# ax[0].axis("image")
ax[1].axis("off")
# the result is a list of the decomposed field.
# each element of the list contains a frame of the decomposition.
# If you want to plot the k-th frame use:
# 1. frame
plot_shifted = True
k_frame = 0
if plot_shifted:
    ax[2].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[2].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    ax[2].pcolormesh(sPOD_frames[k_frame].build_field())
    ax[2].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[2].axis("off")
# ax[1].axis("image")
# 2. frame
k_frame = 1
if plot_shifted:
    im2 = ax[3].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[3].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    im2 = ax[3].pcolormesh(sPOD_frames[k_frame].build_field())
    ax[3].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[3].axis("off")
# ax[2].axis("image")

for axes in ax[:4]:
    axes.set_aspect(0.6)

plt.colorbar(im2)
plt.tight_layout()

# save_fig(pic_dir + "01_traveling_wave_1D_Frames.png",fig)
plt.show()
###########################################


# %%  convergence co-moving ranks
Nx = 400
Nt = Nx // 2  # numer of time intervalls
Niter = 100
case = "multiple_ranks"
fields, shift_list, nmodes, L, dx = generate_data(Nx, Nt, case)
qmat = np.reshape(fields, [Nx, Nt])
data_shape = [Nx, 1, 1, Nt]

transfos = [
    Transform(data_shape, [L], shifts=shift_list[0], dx=[dx], interp_order=3),
    Transform(data_shape, [L], shifts=shift_list[1], dx=[dx], interp_order=3),
]
mu = Nx * Nt / (4 * np.sum(np.abs(qmat)))
lambd0 = 1 / np.sqrt(np.maximum(Nx, Nt))
myparams = sPOD_Param()
myparams.eps = 1e-16
myparams.maxit = Niter
myparams.lambd_s = lambd0
myparams.lambda_E = mu
nmodes_param = np.max(nmodes) + 50

ret = shifted_POD_FB(qmat, transfos, nmodes_param, myparams, method="JFB")

xlims = [-1, Niter]
plt.close(11)
fig, ax = plt.subplots(num=11)
plt.plot(ret.ranks_hist[0], "+", label="$\mathrm{rank}(\mathbf{Q}^1)$")
plt.plot(ret.ranks_hist[1], "x", label="$\mathrm{rank}(\mathbf{Q}^2)$")
plt.plot(xlims, [nmodes[0], nmodes[0]], "k--", label="exact rank $r_1=%d$" % nmodes[0])
plt.plot(xlims, [nmodes[1], nmodes[1]], "k-", label="exact rank $r_2=%d$" % nmodes[1])
plt.xlim(xlims)
plt.xlabel("iterations")
plt.ylabel("rank $r_k$")
plt.legend()

left, bottom, width, height = [0.5, 0.45, 0.3, 0.35]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.pcolormesh(qmat)
ax2.axis("off")
ax2.set_title(r"$\mathbf{Q}$")


# save_fig(pic_dir + "/convergence_ranks_shifted_rPCA.png",fig)

###########################################
# %%  visualize your results: sPOD frames
##########################################
# first we plot the resulting field
sPOD_frames, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
qf = [
    np.squeeze(np.reshape(transfo.apply(frame.build_field()), data_shape))
    for transfo, frame in zip(transfos, ret.frames)
]

gridspec = {"width_ratios": [1, 1, 1, 1]}
fig, ax = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw=gridspec, num=105)
mycmap = "viridis"
vmin = np.min(qtilde) * 0.6
vmax = np.max(qtilde) * 0.6

ax[0].pcolormesh(qmat, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[0].set_title(r"$\mathbf{Q}$")
# ax[0].axis("image")
ax[0].axis("off")

ax[1].pcolormesh(qtilde, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[1].set_title(r"$\tilde{\mathbf{Q}}$")
# ax[0].axis("image")
ax[1].axis("off")
# the result is a list of the decomposed field.
# each element of the list contains a frame of the decomposition.
# If you want to plot the k-th frame use:
# 1. frame
plot_shifted = True
k_frame = 0
if plot_shifted:
    ax[2].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[2].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    ax[2].pcolormesh(sPOD_frames[k_frame].build_field())
    ax[2].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[2].axis("off")
# ax[1].axis("image")
# 2. frame
k_frame = 1
if plot_shifted:
    im2 = ax[3].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[3].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    im2 = ax[3].pcolormesh(sPOD_frames[k_frame].build_field())
    ax[3].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[3].axis("off")
# ax[2].axis("image")

for axes in ax[:4]:
    axes.set_aspect(0.6)

plt.colorbar(im2)
plt.tight_layout()

# save_fig(pic_dir + "multiple_traveling_wave_1D_Frames.png",fig)
plt.show()

####################################################################
# shifted RPC: DATA with noise!!!!
####################################################################
# %%  convergence co-moving ranks
Nx = 400
Nt = Nx // 2  # numer of time intervalls
Niter = 10
case = "sine_waves_noise"
fields, shift_list, nmodes, L, dx = generate_data(Nx, Nt, case)
qmat = np.reshape(fields, [Nx, Nt])
data_shape = [Nx, 1, 1, Nt]

transfos = [
    Transform(data_shape, [L], shifts=shift_list[0], dx=[dx], interp_order=5),
    Transform(data_shape, [L], shifts=shift_list[1], dx=[dx], interp_order=5),
]
mu = Nx * Nt / (4 * np.sum(np.abs(qmat))) * 0.1
lambd0 = 1 / np.sqrt(np.maximum(Nx, Nt))
ret_E = shifted_POD_ALM(
    qmat,
    transfos,
    nmodes_max=np.max(nmodes) + 50,
    eps=1e-16,
    Niter=Niter,
    use_rSVD=False,
    lambd=lambd0,
    mu=mu,
)
iter_ranks = [
    i
    for i in range(len(ret_E.ranks_hist[0]))
    if ret_E.ranks_hist[0][i] == nmodes[0] and ret_E.ranks_hist[1][i] == nmodes[1]
][0] + 1
ret = shifted_POD_ALM(
    qmat,
    transfos,
    nmodes_max=np.max(nmodes) + 50,
    eps=1e-16,
    Niter=Niter,
    use_rSVD=False,
    lambd=lambd0 * 1000,
    mu=mu,
)


xlims = [-1, Niter]
plt.close(14)
fig, ax = plt.subplots(num=14)
handl_list = []
h = plt.plot(
    ret.ranks_hist[0], "o:", fillstyle="none", label="$\mathrm{rank}(\mathbf{Q}^1)$"
)

h = plt.plot(
    ret.ranks_hist[1], "+:", fillstyle="none", label="$\mathrm{rank}(\mathbf{Q}^2)$"
)
h = plt.plot(ret_E.ranks_hist[0], "<:", label="$\mathrm{rank}(\mathbf{Q}^1)$ robust")
h = plt.plot(ret_E.ranks_hist[1], "x:", label="$\mathrm{rank}(\mathbf{Q}^2)$ robust")
h = plt.plot(
    xlims, [nmodes[0], nmodes[0]], "k--", label="exact rank $r_1=%d$" % nmodes[0]
)
h = plt.plot(
    xlims, [nmodes[1], nmodes[1]], "k-", label="exact rank $r_2=%d$" % nmodes[1]
)
plt.xlim(xlims)
plt.xlabel("iterations")
plt.ylabel("rank $r_k$")

plt.legend()
#
# legend1 = plt.legend(handles=handl_list[:2], title="not noise robust",
#                     loc='upper left', bbox_to_anchor=(0.65, 0.9), fontsize='small', frameon=False)
# legend2 = plt.legend(handles=handl_list[2:4], title="noise robust",
#                     loc='upper left', bbox_to_anchor=(0.65, 0.7), fontsize='small', frameon=False)
# legend = plt.legend(handles=handl_list[4:],
#                     loc='upper left', bbox_to_anchor=(0.65, 0.5), fontsize='small', frameon=False)
# ax.add_artist(legend1)
# ax.add_artist(legend2)
#
# left, bottom, width, height = [0.5, 0.45, 0.3, 0.35]
# ax2 = fig.add_axes([left, bottom, width, height])
# ax2.pcolormesh(qmat)
# ax2.axis("off")
# ax2.set_title(r"$\mathbf{Q}$")


# save_fig(pic_dir + "/convergence_ranks_shifted_rPCA_noise.png",fig)

###########################################
# %%  visualize your results: sPOD frames
##########################################
# facecolors='none' first we plot the resulting field
Niter = 500
ret_E = shifted_POD_ALM(
    qmat,
    transfos,
    nmodes_max=np.max(nmodes) + 50,
    eps=1e-16,
    Niter=Niter,
    use_rSVD=False,
    lambd=lambd0,
    mu=mu,
)

sPOD_frames, qtilde, rel_err = ret_E.frames, ret_E.data_approx, ret.rel_err_hist
qf = [
    np.squeeze(np.reshape(transfo.apply(frame.build_field()), data_shape))
    for transfo, frame in zip(transfos, ret_E.frames)
]

gridspec = {"width_ratios": [1, 1, 1, 1, 1]}
fig, ax = plt.subplots(1, 5, figsize=(12, 4), gridspec_kw=gridspec, num=105)
mycmap = "viridis"
vmin = np.min(qtilde) * 0.6
vmax = np.max(qtilde) * 0.6

ax[0].pcolormesh(qmat, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[0].set_title(r"$\mathbf{Q}$")
# ax[0].axis("image")
ax[0].axis("off")

ax[1].pcolormesh(qtilde, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[1].set_title(r"$\tilde{\mathbf{Q}}$")
# ax[0].axis("image")
ax[1].axis("off")
# the result is a list of the decomposed field.
# each element of the list contains a frame of the decomposition.
# If you want to plot the k-th frame use:
# 1. frame
plot_shifted = True
k_frame = 0
if plot_shifted:
    ax[2].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[2].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    ax[2].pcolormesh(sPOD_frames[k_frame].build_field())
    ax[2].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[2].axis("off")
# ax[1].axis("image")
# 2. frame
k_frame = 1
if plot_shifted:
    im2 = ax[3].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[3].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    im2 = ax[3].pcolormesh(sPOD_frames[k_frame].build_field())
    ax[3].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[3].axis("off")
# ax[2].axis("image")

# 3. noise
im2 = ax[4].pcolormesh(ret_E.error_matrix, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[4].set_title(r"$\mathbf{E}$")
ax[4].axis("off")

for axes in ax[:5]:
    axes.set_aspect(0.6)

plt.colorbar(im2)
plt.tight_layout()

# save_fig(pic_dir + "traveling_waves_noise_1D_Frames.png",fig)
plt.show()

# %% compare shifted rPCA and shifted POD
Niter = 500
Nx = 400
Nt = Nx // 2
case = "multiple_ranks"
fields, shift_list, nmodes, L, dx = generate_data(Nx, Nt, case, noise_percent=0.125)
qmat = np.reshape(fields, [Nx, Nt])
data_shape = [Nx, 1, 1, Nt]

linestyles = ["--", "-.", ":", "-", "-."]
plot_list = []
mu0 = Nx * Nt / (4 * np.sum(np.abs(qmat)))
lambd0 = 1 / np.sqrt(np.maximum(Nx, Nt))
ret_list = []
plt.close(87)
fig, ax = plt.subplots(num=87)
for ip, fac in enumerate([0.0001, 0.1, 1, 10, 1000]):  # ,400, 800]):#,800,1000]:
    mu = mu0 * fac
    # transformations with interpolation order T^k of Ord(h^5) and T^{-k} of Ord(h^5)
    transfos = [
        Transform(data_shape, [L], shifts=shift_list[0], dx=[dx], interp_order=[5, 5]),
        Transform(data_shape, [L], shifts=shift_list[1], dx=[dx], interp_order=[5, 5]),
    ]

    ret = shifted_POD_ALM(
        qmat,
        transfos,
        nmodes_max=np.max(nmodes) + 10,
        eps=1e-16,
        Niter=Niter,
        use_rSVD=True,
        lambd=lambd0,
        mu=mu,
    )

    ret_list.append(ret)
    h = ax.semilogy(
        np.arange(0, np.size(ret.rel_err_hist)),
        ret.rel_err_hist,
        linestyles[ip],
        label="sPOD-$\mathcal{J}_1$ $\lambda=10^{%d}\lambda_0$" % int(np.log10(fac)),
    )
    plt.text(
        Niter,
        ret.rel_err_hist[-1],
        "$(r_1,r_2)=(%d,%d)$" % (ret.ranks[0], ret.ranks[1]),
        transform=ax.transData,
        va="bottom",
        ha="right",
    )
    # plot_list.append(h)
# sPOD results           Xgrid - 0.1 * L) / w)

ret = shifted_POD_J2(qmat, transfos, nmodes=nmodes, eps=1e-16, Niter=Niter)
# %
h = ax.semilogy(
    np.arange(0, np.size(ret.rel_err_hist)),
    ret.rel_err_hist,
    "k-",
    label="sPOD-$\mathcal{J}_2$ $(r_1,r_2)=(%d,%d)$" % (nmodes[0], nmodes[1]),
)
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower right")
plt.subplots_adjust(bottom=0.2, top=0.8)
plt.tight_layout(pad=3.0)
ax.set_xlim(-5, ax.get_xlim()[-1])
plt.ylabel(r"relative error")
plt.xlabel(r"iteration")
# save_fig(pic_dir + "/convergence_J1_vs_J2_noise.png",fig)
plt.show()


###########################################

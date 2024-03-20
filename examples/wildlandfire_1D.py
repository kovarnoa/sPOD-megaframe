import sys

sys.path.append("../lib")
import numpy as np
from numpy import meshgrid
import matplotlib.pyplot as plt
from sPOD_algo import (
    shifted_POD_FB,
    shifted_POD_ALM,
    sPOD_Param,
    give_interpolation_error,
)
from transforms import Transform
from plot_utils import save_fig

pic_dir = pic_dir = "../images/"


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
    [X, T] = meshgrid(x, t)
    X = X.T
    T = T.T
    dx = x[1] - x[0]
    nmodes = [1, 1, 1]

    return Q, shift_list, L, dx, Nx, Nt, nmodes


Niter = 400

prefix = "Wildlandfire_1d"
# prefix = "small_wildlandfire"

fields, shift_list, L, dx, Nx, Nt, nmodes = generate_wildlandfire_data(
    f"../examples/{prefix}/1D_Grid.npy",
    f"../examples/{prefix}/Time.npy",
    f"../examples/{prefix}/SnapShotMatrix558.49.npy",
    f"../examples/{prefix}/Shifts558.49.npy",
)

data_shape = [Nx, 1, 1, Nt]
trafos = [
    Transform(data_shape, [L], shifts=shift_list[0], dx=[dx], interp_order=5),
    Transform(data_shape, [L], shifts=shift_list[1], dx=[dx], interp_order=5),
    Transform(data_shape, [L], shifts=shift_list[2], dx=[dx], interp_order=5),
]

interp_err = np.max([give_interpolation_error(fields, trafo) for trafo in trafos])
print("interpolation error: %1.2e " % interp_err)

qmat = np.reshape(fields, [Nx, Nt])

method = "shifted_POD_ALM"
# method = "shifted_POD_JFB"
# method = "shifted_POD_BFB"
lambda0 = 4000  # for Temperature
# lambda0 = 27  # for supply mass
myparams = sPOD_Param()
myparams.maxit = Niter

if method == "shifted_POD_ALM":
    ret = shifted_POD_ALM(
        qmat,
        trafos,
        myparams,
        use_rSVD=True,
        lambd=1 / np.sqrt(np.maximum(Nx, Nt)) * 1,
        mu=Nx * Nt / (4 * np.sum(np.abs(qmat))) * 0.01,
    )
elif method == "shifted_POD_BFB":
    myparams = sPOD_Param(
        maxit=Niter,
        lambda_s=lambda0,
        total_variation_iterations=40,
    )
    ret = shifted_POD_FB(qmat, trafos, nmodes, myparams, method="BFB")
elif method == "shifted_POD_JFB":
    myparams = sPOD_Param(
        maxit=Niter,
        lambda_s=lambda0,
        total_variation_iterations=40,
    )
    ret = shifted_POD_FB(qmat, trafos, nmodes, myparams, method="JFB")


sPOD_frames, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
qf = [
    np.squeeze(np.reshape(trafo.apply(frame.build_field()), data_shape))
    for trafo, frame in zip(trafos, ret.frames)
]


# %% 1. visualize your results: sPOD frames
##########################################
# first we plot the resulting field
gridspec = {"width_ratios": [1, 1, 1, 1, 1]}
fig, ax = plt.subplots(1, 5, figsize=(12, 5), gridspec_kw=gridspec, num=101)
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
# 3. frame
k_frame = 2
if plot_shifted:
    im2 = ax[4].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[4].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    im2 = ax[4].pcolormesh(sPOD_frames[k_frame].build_field())
    ax[4].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[4].axis("off")
# ax[2].axis("image")

for axes in ax[:4]:
    axes.set_aspect(0.6)

plt.colorbar(im2)
plt.tight_layout()

# save_fig(pic_dir + "wildlandfire_S.png", fig)
plt.show()


# ####### Visualize convergence of co moving ranks ###################
# xlims = [-1, Niter]
# plt.close(11)
# fig, ax = plt.subplots(num=11)
# plt.plot(ret.ranks_hist[0], "+", label="$\mathrm{rank}(\mathbf{Q}^1)$")
# plt.plot(ret.ranks_hist[1], "x", label="$\mathrm{rank}(\mathbf{Q}^2)$")
# plt.plot(ret.ranks_hist[2], "*", label="$\mathrm{rank}(\mathbf{Q}^3)$")
# plt.plot(xlims, [nmodes[0], nmodes[0]], "k--", label="exact rank $r_1=%d$" % nmodes[0])
# plt.plot(xlims, [nmodes[1], nmodes[1]], "k-", label="exact rank $r_2=%d$" % nmodes[1])
# plt.plot(xlims, [nmodes[2], nmodes[2]], "k-.", label="exact rank $r_3=%d$" % nmodes[2])
# plt.xlim(xlims)
# plt.xlabel("iterations")
# plt.ylabel("rank $r_k$")
# plt.legend()

# left, bottom, width, height = [0.5, 0.45, 0.3, 0.35]
# ax2 = fig.add_axes([left, bottom, width, height])
# ax2.pcolormesh(qmat)
# ax2.axis("off")
# ax2.set_title(r"$\mathbf{Q}$")

# plt.show()
# save_fig(pic_dir + "ranks_wildlandfire_S.png", fig)


# %% compare shifted rPCA and shifted POD

linestyles = ["--", "-.", ":", "-", "-."]
plot_list = []
mu0 = Nx * Nt / (4 * np.sum(np.abs(qmat)))
lamb = 4000
ret_list = []
plt.close(87)
fig, ax = plt.subplots(num=87)
for ip, fac in enumerate([0.01, 0.1, 0, 10, 100]):  # ,400, 800]):#,800,1000]:
    lambd = lamb * fac
    # transformations with interpolation order T^k of Ord(h^5) and T^{-k} of Ord(h^5)
    if method == "shifted_POD_JFB":
        myparams = sPOD_Param(maxit=Niter, lambda_s=lambd)
        ret = shifted_POD_FB(qmat, trafos, nmodes, myparams, method="JFB")
    elif method == "shifted_POD_BFB":
        myparams = sPOD_Param(maxit=Niter, lambda_s=lambd)
        ret = shifted_POD_FB(qmat, trafos, nmodes, myparams, method="BFB")

    ret_list.append(ret)
    h = ax.semilogy(
        np.arange(0, np.size(ret.rel_err_hist)),
        ret.rel_err_hist,
        linestyles[ip],
        label="sPOD_BFB-$\mathcal{J}_1$ $\lambda_{\sigma}^0=10^{%d}+\lambda_{\sigma}$"
        % int(np.log10(fac)),
    )
    plt.text(
        Niter,
        ret.rel_err_hist[-1],
        "$(r_1,r_2,r_3)=(%d,%d,%d)$" % (ret.ranks[0], ret.ranks[1], ret.ranks[2]),
        transform=ax.transData,
        va="bottom",
        ha="right",
    )


# plt.text(Niter, ret.rel_err_hist[-1], "$(r_1,r_2)=(%d,%d)$" % (nmodes[0], nmodes[1]), transform=ax.transData,
#         va='bottom', ha="right")
# interp_err = np.max([give_interpolation_error(fields, trafo) for trafo in trafos])
# ax.hlines(interp_err, -5, Niter + 5, "k",linewidth=5,alpha = 0.3)
# plt.text(10,interp_err,"$\mathcal{E}_*$",transform=ax.transData ,  va='bottom', ha="left")
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower right")
plt.subplots_adjust(bottom=0.2, top=0.8)
plt.tight_layout(pad=3.0)
ax.set_xlim(-5, ax.get_xlim()[-1])
plt.ylabel(r"relative error")
plt.xlabel(r"iteration")
# save_fig(pic_dir + "/convergence_wildlandfire_S_sPOD_DFB.png", fig)
plt.show()

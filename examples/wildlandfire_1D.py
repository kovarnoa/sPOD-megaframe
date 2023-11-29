import sys

sys.path.append("../lib")
import numpy as np
from numpy import meshgrid
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sPOD_tools import (
    shifted_rPCA,
    shifted_POD,
    shifted_POD_BFB,
    shifted_POD_ADM,
    sPOD_Param,
    give_interpolation_error,
)
from transforms import transforms
from plot_utils import save_fig


def generate_wildlandfire_data(grid, time, snapshot, shifts):
    L = 1
    x = np.load(grid, allow_pickle=True)[0]
    t = np.load(time)
    Nx = len(x)
    Nt = len(t)
    q = np.load(snapshot)
    q1 = q[:Nx, :]
    q2 = q[Nx:, :]
    Q = q1 + q2
    shift_list = np.load(shifts)
    [X, T] = meshgrid(x, t)
    X = X.T
    T = T.T
    dx = x[1] - x[0]
    nmodes = [4, 4, 4]

    return Q, shift_list, L, dx, Nx, Nt, nmodes


Niter = 2

fields, shift_list, L, dx, Nx, Nt, nmodes = generate_wildlandfire_data(
    "../examples/Wildlandfire_1d/1D_Grid.npy",
    "../examples/Wildlandfire_1d/Time.npy",
    "../examples/Wildlandfire_1d/SnapShotMatrix558.49.npy",
    "../examples/Wildlandfire_1d/Shifts558.49.npy",
)

data_shape = [Nx, 1, 1, Nt]
trafos = [
    transforms(data_shape, [L], shifts=shift_list[0], dx=[dx], interp_order=5),
    transforms(data_shape, [L], shifts=shift_list[1], dx=[dx], interp_order=5),
    transforms(data_shape, [L], shifts=shift_list[2], dx=[dx], interp_order=5),
]

interp_err = np.max([give_interpolation_error(fields, trafo) for trafo in trafos])
print("interpolation error: %1.2e " % interp_err)

qmat = np.reshape(fields, [Nx, Nt])

myparams = sPOD_Param(
    maxit=Niter,
    lamb=Nx * Nt / (4 * np.sum(np.abs(qmat))),
    total_variation_iterations=40,
)
ret = shifted_POD_BFB(qmat, trafos, np.max(nmodes), myparams)

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

# save_fig(pic_dir + "01_traveling_wave_1D_Frames.png", fig)
plt.show()

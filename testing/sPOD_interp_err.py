import sys
sys.path.append('../lib/')

import numpy as np
from numpy import mod, exp
from plot_utils import save_fig
from numpy.linalg import norm
from sPOD_tools import shifted_rPCA, shifted_POD
from transforms import transforms
import matplotlib.pyplot as plt

# %% parameters
L      = 1
Tmax   = 0.5                        # final time
M      = 400                        # Number of grid points
Nt = 200                            # Number of time steps
x = np.arange(0,M)/M*L
t =np.arange(0,Nt)/Nt*Tmax
[X, T]= np.meshgrid(x,t)            # X-T Grid
X = X.T
T = T.T
Maxiter = 1000
delta  = 0.0125
lambd = 0.1 * L
sigma = 0.015*L  # standard diviation of the pulse
dx = x[1]-x[0]
dt = t[1]-t[0]

############################
# %% DATA Generation
############################

init_case = "one-transport"

if init_case == "one-transport":
    c = 1
    fun = lambda x, shifts, t: exp(-(mod((x + shifts[0]), L) - 0.1) ** 2 / sigma ** 2) #+ \
    fun_reference_frame = lambda x, t: exp(-(mod((x), L) - 0.1) ** 2 / sigma ** 2)  # + \
                               #exp(-(mod((x + shifts[1]), L) - 0.9) ** 2 / sigma ** 2)
    dfun = lambda x, shifts, t: -exp(-(mod((x + shifts[0]), L) - 0.1) ** 2 / sigma ** 2)  # + \
    d4fun = lambda x, shifts, t: 12*exp(-(mod((x + shifts[0]), L) - 0.1)**2/sigma**2)/sigma**4 - 48*(mod((x + shifts[0]), L) - 0.1)**2*exp(-(mod((x + shifts[0]), L) - 0.1)**2/sigma**2)/sigma**6 + 16*(mod((x + shifts[0]), L) - 0.1)**4*exp(-(mod((x + shifts[0]), L) - 0.1)**2/sigma**2)/sigma**8
    d2fun = lambda x, shifts, t: -2*exp(-(mod((x + shifts[0]), L) - 0.1)**2/sigma**2)/sigma**2 + 4*(mod((x + shifts[0]), L) - 0.1)**2*exp(-(mod((x + shifts[0]), L) - 0.1)**2/sigma**2)/sigma**4

    # Define your field as a list of fields:
    # For example the first element in the list can be the density of
    # a flow quantity and the second element could be the velocity in 1D
    shifts = [np.asarray([-c * t]), np.asarray([c * t])]

    density = fun(X, shifts, T)
    velocity = fun(X, shifts, T)
    fields = density
    fields_ref = fun_reference_frame(X,T)

######################################
# %% CALL THE SPOD algorithm
######################################
err_list = []
err_list_1st_ord = []
spacing_list =[]
for M in np.logspace(1.5,5.1,10,dtype=np.int32):
    x = np.arange(0, M) / M * L
    dx = x[1] - x[0]
    [X, T] = np.meshgrid(x, t)
    X = X.T
    T = T.T

    fields = fun(X, shifts, T)
    qmat = np.reshape(fields,[M,Nt])
    fields_ref = fun_reference_frame(X, T)
    qmat_ref = np.reshape(fields_ref, [M, Nt])
    data_shape = [M,1,1,Nt]
    trafos = [transforms(data_shape ,[L], shifts = shifts[0].flatten(), dx = [dx] , use_scipy_transform=False, interp_order=3),
                transforms(data_shape ,[L], shifts = shifts[1].flatten(), dx = [dx] , use_scipy_transform=False, interp_order=3)]

    trafos_1ord = [transforms(data_shape ,[L], shifts = shifts[0].flatten(), dx = [dx] , use_scipy_transform=False, interp_order=5),
                transforms(data_shape ,[L], shifts = shifts[1].flatten(), dx = [dx] , use_scipy_transform=False, interp_order=5)]
    #plt.pcolormesh(X,T,qmat)
    err = np.max(np.abs(qmat_ref- trafos[0].reverse(qmat)))#/norm(qmat)
    err_list.append(err)

    err = np.max(np.abs(qmat_ref - trafos_1ord[0].reverse(qmat)))  # /norm(qmat)
    err_list_1st_ord.append(err)
    spacing_list.append(dx)

# %% plot err
plt.loglog(spacing_list,err_list,'-.*',label="interp. $n=3$")
plt.loglog(spacing_list,err_list_1st_ord,':o',label="interp. $n=1$")
err_fun3 = lambda h,qmat: 9/16*h**4*np.max(np.abs(d4fun(X,shifts,T)))/(4*3*2)
err_fun1 = lambda h,qmat: 1/4*h**2*np.max(np.abs(d2fun(X,shifts,T)))/(2)
plt.loglog(spacing_list,err_fun3(np.asarray(spacing_list),qmat),"k-.", label="bound $n=3$")
plt.loglog(spacing_list,err_fun1(np.asarray(spacing_list),qmat),"k:", label="bound $n=1$")
plt.xlabel("$h$")
plt.ylabel("$\Vert \mathbf{E} \Vert_\infty$")
plt.legend(fontsize="small")
plt.grid(which="both",linestyle=':')
plt.show()
save_fig("images/transform_error.png")
# plt.pcolormesh(X,T,qmat - trafos[0].apply(trafos[0].reverse(qmat)))
# plt.colorbar()





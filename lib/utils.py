import numpy as np
import scipy.sparse as sp


def derivative(N, h, coefficient, boundary="periodic"):
    """
    Compute the discrete derivative for periodic BC
    """
    dlow = -(np.size(coefficient) - 1) // 2
    dup = - dlow+1

    diagonals = []
    offsets = []
    for k in np.arange(dlow, dup):
        diagonals.append(coefficient[k - dlow] * np.ones(N - abs(k)))
        offsets.append(k)
        if k > 0:
            diagonals.append(coefficient[k - dlow] * np.ones(abs(k)))
            offsets.append(-N + k)
        if k < 0:
            diagonals.append(coefficient[k - dlow] * np.ones(abs(k)))
            offsets.append(N + k)

    return sp.diags(diagonals, offsets) / h


class finite_diffs:
    def __init__(self, Ngrid, dX):
        Ix = sp.eye(Ngrid[0])
        Iy = sp.eye(Ngrid[1])
        # stencilx = np.asarray( [-1/60,	3/20, 	-3/4, 	0, 	3/4, 	-3/20, 	1/60])
        stencil_x = np.asarray([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
        # stencil_x = np.asarray([-0.5,0,0.5])
        # stencil_xx = np.asarray([1/90,	-3/20, 	3/2, 	-49/18, 	3/2, 	-3/20, 	1/90])
        stencil_xx = np.asarray([-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560])
        self.Dx_mat = sp.kron(derivative(Ngrid[0], dX[0], stencil_x), Iy)
        self.Dy_mat = sp.kron(Ix, derivative(Ngrid[1], dX[1], stencil_x))
        self.Dxx_mat = sp.kron(derivative(Ngrid[0], dX[0] ** 2, stencil_xx), Iy)
        self.Dyy_mat = sp.kron(Ix, derivative(Ngrid[1], dX[1] ** 2, stencil_xx))


    def Dx(self, q):
        input_shape = np.shape(q)
        q = np.reshape(q,-1)
        return np.reshape(self.Dx_mat @ q, input_shape)
    def Dxx(self, q):
        input_shape = np.shape(q)
        q = np.reshape(q, -1)
        return np.reshape(self.Dxx_mat @ q, input_shape)
    def Dy(self, q):
        input_shape = np.shape(q)
        q = np.reshape(q, -1)
        return np.reshape(self.Dy_mat @ q, input_shape)
    def Dyy(self, q):
        input_shape = np.shape(q)
        q = np.reshape(q, -1)
        return np.reshape(self.Dyy_mat @ q, input_shape)
    def rot(self, v1, v2):
        return self.Dx(v2) - self.Dy(v1)



def calculate_forces(uxs, uys, chi, dx, dy):
    """
    uxs: relativ x - velocity u_fluid - u_solid
    uys: relativ y - velocity u_fluid - u_solid
    chi: mask function multiplied by the penalisation constant
    dx, dy: lattice spacings
    """

    forcex = np.sum(np.sum(uxs*chi)) * dx * dy
    frocey = np.sum(np.sum(uys*chi)) * dx * dy

    return forcex, frocey




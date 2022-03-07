

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:31:40 2021

@author: Philipp Krah

 CLASS of Transformation applied to the field

"""
from scipy import sparse
from numpy import exp, mod, meshgrid, pi, sin ,size, reshape
import numpy as np
import scipy.ndimage as ndimage
# %%


# %%
#########################
# Lagrange interpolation
#########################
def lagrange(xvals, xgrid, j):
    """
    Returns the j-th basis polynomial evaluated at xvals
    using the grid points listed in xgrid
    """
    xgrid = np.asarray(xgrid)
    if not isinstance(xvals,list):
        xvals=[xvals]
    n = np.size(xvals)
    Lj = np.zeros(n)
    for i,xval in enumerate([xvals]):
        nominator = xval - xgrid
        denominator = xgrid[j] - xgrid
        p = nominator/(denominator+1e-32)                               #add SMALL for robustness
        p[j] = 1
        Lj[i] = np.prod(p)

    return Lj


class transforms:
    # TODO: Add properties of class frame in the description
    """ Class of all Transforms.
        A transformation can be implemented as a
            + shift T^c q(x,t) = q(x-ct,t)
            + rotation T^w q(x,y,t) = q(R(omega)(x,y),t) where R(omega) is the rotation matrix
    """

    def __init__(self, data_shape, domain_size, trafo_type="shift", shifts = None, \
                 dx = None, rotations=None, rotation_center = None, use_scipy_transform = True):
        self.Ngrid = data_shape[:2]
        self.Nvar = data_shape[2]
        self.Ntime = data_shape[3]
        self.data_shape = data_shape
        self.domain_size = domain_size
        self.trafo_type = trafo_type
        self.dim = size(dx)
        self.dx = dx  # list of lattice spacings
        if self.dim == 1:
            self.shifts_pos, self.shifts_neg = self.init_shifts_1D(dx[0], domain_size[0], self.Ngrid[0], shifts[0,:], Nvar=self.Nvar)
            self.shift = self.shift1
        else:
            if trafo_type=="shift":
                if use_scipy_transform:
                    self.shift = self.shift_scipy
                    self.shifts_pos =  shifts    # dim x Ntime shiftarray (one element for one time instance)
                    one = np.ones_like(shifts)
                    one[0,:]=dx[0]
                    one[1,:]=0#dx[1]
                    self.shifts_neg = -shifts  # dim x Ntime shiftarray (one element for one time instance)
                else: # own implementation for shifts: is much faster then ndimage
                    self.shifts_pos, self.shifts_neg = self.init_shifts_2D(dx, domain_size, self.Ngrid, shifts, Nvar= self.Nvar)
                    self.shift = self.shift1
            if trafo_type=="rotation":
                assert(size(dx)==2), "is only implemented for spatial fields in 2 dimensions"
                #assert(np.sum(rotation_center)==0), "rotation center should be in the middle!"
                self.rotations = rotations # is an array with [omega(t_1), ... omega(t_Ntime)] rotation angles at different timepoints
                self.rotation_center = rotation_center # array including center of rotation(x_0, y_0)
            if trafo_type=="shiftRot":
                assert(size(dx)==2), "is only implemented for spatial fields in 2 dimensions"
                if use_scipy_transform:
                    self.shift = self.shift_scipy
                    self.shifts_pos =  shifts    # dim x Ntime shiftarray (one element for one time instance)
                    self.shifts_neg = -shifts  # dim x Ntime shiftarray (one element for one time instance)
                else: # own implementation for shifts: is much faster then ndimage
                    self.shifts_pos, self.shifts_neg = self.init_shifts_2D(dx, domain_size, self.Ngrid, shifts, Nvar= self.Nvar)
                    self.shift = self.shift1
                self.rotations = rotations
                self.rotation_center = rotation_center



    def apply(self, field):
        """
        This function returns the shifted field.
        $q(x-s,t)=T^s[q(x,t)]$
        here the shift is simply s=c*t
        In the default case where c= ~velocity of the frame~,
        the field is shifted back to the original frame.
        ( You may call it the labratory frame)
        Before we shift the frame has to be put togehter in the co-moving
        frame. This can be done by build_field().

        """
        input_shape = np.shape(field)
        field = reshape(field,self.data_shape)
        if self.trafo_type=="shift":
            ftrans = self.shift(field, self.shifts_pos)
        elif self.trafo_type == "rotation":
            ftrans = self.rotate(field, self.rotations)
        elif self.trafo_type == "shiftRot":
            # ~ auxField = self.shift(field,self.shiftMatrices_pos)         #shift to origin
            field = self.rotate(field,self.rotations)                 #rotate and return
            ftrans = self.shift(field,self.shifts_pos)                   #shift to origin
        elif self.trafo_type == "identity":
            ftrans = field
        else:
            print("Transformation type: %s not known"%self.trafo_type)

        return reshape(ftrans, input_shape)

    def reverse(self, field):
        """
        This function returns the shifted field.
        $q(x-s,t)=T^s[q(x,t)]$
        here the shift is simply s=c*t
        In the default case where c= ~velocity of the frame~,
        the field is shifted back to the original frame.
        ( You may call it the labratory frame)
        Before we shift the frame has to be put togehter in the co-moving
        frame. This can be done by build_field().
        """
        input_shape = np.shape(field)
        field = reshape(field, self.data_shape)
        if self.trafo_type=="shift":
            ftrans = self.shift(field,self.shifts_neg)
        elif self.trafo_type == "rotation":
            ftrans = self.rotate(field, -self.rotations)
        elif self.trafo_type == "shiftRot":
            field = self.shift(field,self.shifts_neg)                   #shift back and return
            ftrans = self.rotate(field,-self.rotations)               #rotate back
            # ~ return self.shift(auxField,self.shiftMatrices_neg)          #shift back and return

        elif self.trafo_type == "identity":
            ftrans = field
        else:
            print("Transformation type: %s not known"%self.trafo_type)

        return reshape(ftrans, input_shape)

    def shift1(self, field, shifts):
        """
        This function returns the shifted field.
        $q(x-s,t)=T^s[q(x,t)]$
        here the shift is simply s=c*t
        In the default case where c= ~velocity of the frame~,
        the field is shifted back to the original frame.
        ( You may call it the labratory frame)
        Before we shift the frame has to be put togehter in the co-moving
        frame. This can be done by build_field().
        
        """
        Ntime = np.size(field,-1)
        field_shift = np.zeros_like(field)
        for it in range(Ntime):
            vec = np.reshape(field[...,it],-1)
            vec_shift = shifts[it]@vec
            field_shift[...,it] = np.reshape(vec_shift,self.data_shape[:-1])

        return field_shift
        
    def shift_scipy(self,field,shifts):
        """
        This function returns the shifted field.
        $q(x-s,t)=T^s[q(x,t)]$ 
        here the shift is simply s=c*t
        In the default case where c= ~velocity of the frame~,
        the field is shifted back to the original frame. 
        ( You may call it the labratory frame)
        Before we shift the frame has to be put togehter in the co-moving
        frame. This can be done by build_field().
        
        """
        input_shape = np.shape(field)
        Ntime = np.size(field,-1)
        field_shift = np.zeros([*self.Ngrid,Ntime])
        for it in range(Ntime):

            #DeltaS = -np.round(np.divide(shifts[:, it], self.dx))
            DeltaS = -np.divide(shifts[:, it], self.dx)
            q = np.reshape(field[...,it], self.Ngrid)
            #field_shift[...,it] = np.roll(q, int(DeltaS[1]), axis=1)
            #field_shift[..., it] = np.roll(q, DeltaS[0], axis=1)
            #q = np.reshape(field[...,it], self.Ngrid)
            field_shift[...,it] = ndimage.shift(q,DeltaS,mode='grid-wrap')
        return np.reshape(field_shift,input_shape)
    # Note (MI): the shifts need to be scaled w.r.t the image and are
    #            reversed 
    
    def rotate(self, field, rotations):
        
        input_shape = np.shape(field)
        Ntime = np.size(field,-1)
        field_rot = np.zeros([*self.Ngrid,Ntime])
        for it in range(Ntime):
            angle = rotations[it]
            q = np.reshape(field[...,it], self.Ngrid)
            field_rot[...,it] = ndimage.rotate(q, angle*180.0/np.pi, reshape=False)
            
        return np.reshape(field_rot,input_shape)
            
            
            
    def init_shifts_2D(self, dX, domain_size, Ngrid, shifts, Nvar = 1):
        ### implement pos shift matrix ###
        shift_pos_mat_list = []
                
        Nx, Ny = Ngrid
        Lx, Ly = domain_size
        dx, dy = dX
        Nt = np.size(shifts,-1)

        if np.ndim(shifts) ==2:
            shiftx_pos = np.zeros([Ny,Nt])
            shifty_pos = np.zeros([Nx, Nt])
            for it in range(Nt):
                shiftx_pos[:, it] = shifts[0, it]
                shifty_pos[:, it] = shifts[1, it]
        else:
            shiftx_pos = shifts[0,...]
            shifty_pos = shifts[1,...]

        
        shiftx_pos_mat_list=self.compute_general_shift_matrix(shiftx_pos, Lx, dx, Nx)
        shifty_pos_mat_list=self.compute_general_shift_matrix(shifty_pos, Ly, dy, Ny)
        
        # kron for each time slice
        for shiftx,shifty in zip(shiftx_pos_mat_list,shifty_pos_mat_list):
            shift_pos_mat_list.append(sparse.kron(sparse.kron(shiftx, shifty), sparse.eye(Nvar)))
        
        ### implement neg shift matrix ###
        shift_neg_mat_list = []
        
        shiftx_neg = -shiftx_pos
        shifty_neg = -shifty_pos
        shiftx_neg_mat_list=self.compute_general_shift_matrix(shiftx_neg, Lx, dx, Nx)
        shifty_neg_mat_list=self.compute_general_shift_matrix(shifty_neg, Ly, dy, Ny)
        
        # kron for each time slice
        for shiftx,shifty in zip(shiftx_neg_mat_list,shifty_neg_mat_list):
            shift_neg_mat_list.append(sparse.kron(sparse.kron(shiftx, shifty), sparse.eye(Nvar)))
        
        return shift_pos_mat_list, shift_neg_mat_list

    def init_shifts_1D(self, dx, Lx, Nx, shifts, Nvar=1):


        ### implement pos shift matrix ###
        shift_pos_mat_list = []
        shiftx_pos = shifts[:]
        shiftx_pos_mat_list = self.compute_shift_matrix(shiftx_pos, Lx, dx, Nx)
        # kron for each time slice
        for shiftx in shiftx_pos_mat_list:
            shift_pos_mat_list.append(sparse.kron(shiftx, sparse.eye(Nvar)))

        ### implement neg shift matrix ###
        shift_neg_mat_list = []
        shiftx_neg = -shiftx_pos
        shiftx_neg_mat_list = self.compute_shift_matrix(shiftx_neg, Lx, dx, Nx)

        # kron for each time slice
        for shiftx in shiftx_neg_mat_list:
            shift_neg_mat_list.append(sparse.kron(shiftx, sparse.eye(Nvar)))

        return shift_pos_mat_list, shift_neg_mat_list
    
    def init_rotations(self):
        ### implement pos shift matrix ###
        rotation_pos_mat_list = []
                
        Nx, Ny = self.Ngrid
        Lx, Ly = self.domain_size
        dx, dy = self.dx
        
        rotations_pos = self.rotations
        rot_center    = self.rotation_center
        rotations_neg = - rotations_pos
        
        rotation_pos_mat_list=self.compute_rotation_matrix(rotations_pos, rot_center, \
                                                         self.domain_size, self.dx, self.Ngrid)
        
        rotation_neg_mat_list=self.compute_rotation_matrix(rotations_neg, rot_center, \
                                                         self.domain_size, self.dx, self.Ngrid)
        
        return rotation_pos_mat_list, rotation_neg_mat_list
        
    def compute_shift_matrix(self,shift_list, domain_length, spacing, Npoints):
        from numpy import floor
        
        Mat = []
        for shift in shift_list:
            
            # we assume periodicity here
            if shift > domain_length:
                shift = shift - domain_length
            elif shift < 0:
                shift = shift + domain_length
        
            ''' interpolation scheme        lagrange_idx(x)= (x-x_{idx-1})/(x_idx - x_0)+
            -1      0   x    1       2                    ...+(x-x_{idx+2})/(x_idx - x_{idx+2})
             +      +   x    +       +
           idx_m1  idx_0    idx_1   idx_2
          =idx_0-1        =idx_0+1
           '''
           
            # shift is close to some discrete index:
            idx_0 = floor(shift/spacing)
            # save all neighbours
            idx_list = np.asarray([idx_0-1, idx_0, idx_0+1, idx_0+2],dtype=np.int32)
            
            if idx_list[0] < 0 : idx_list[0] += Npoints
            if idx_list[2] > Npoints - 1: idx_list[2] -= Npoints
            if idx_list[3] > Npoints - 1: idx_list[3] -= Npoints
            # subdiagonals needed if point is on other side of domain
            idx_subdiags_list = idx_list - Npoints
            # compute the distance to the index
            delta_idx = shift/spacing - idx_0
            # compute the 4 langrage basis elements
            lagrange_coefs = [lagrange(delta_idx, [-1,0,1,2], j) for j in range(4)]
            # for the subdiagonals as well
            lagrange_coefs = lagrange_coefs + lagrange_coefs
            
            # band diagonals for the shift matrix
            offsets = np.concatenate([idx_list,idx_subdiags_list])
            diagonals = [np.ones(Npoints)*Lj  for Lj in lagrange_coefs]

            Mat.append(sparse.diags(diagonals,offsets,shape=[Npoints,Npoints]))
        
        return Mat

    def compute_general_shift_matrix(self, shifts, domain_length, spacing, Npoints):
        """

        :param shifts: shift(x_i,t_j) assumes an array of size i=0,...,Nx -1 ; j=0,...,Nt
        :param domain_length:
        :param spacing:
        :param Npoints:
        :return:
        """
        from numpy import floor

        Nx, Nt = np.shape(shifts)
        Mat = []
        for it in range(Nt):
            col = []
            row = []
            val = []
            for ix in range(Nx):
                shift = shifts[ix,it]
                # we assume periodicity here
                if shift > domain_length:
                    shift = shift - domain_length
                elif shift < 0:
                    shift = shift + domain_length

                ''' interpolation scheme        lagrange_idx(x)= (x-x_{idx-1})/(x_idx - x_0)+
                -1      0   x    1       2                    ...+(x-x_{idx+2})/(x_idx - x_{idx+2})
                 +      +   x    +       +
               idx_m1  idx_0    idx_1   idx_2
              =idx_0-1        =idx_0+1
               '''

                # shift is close to some discrete index:
                idx_0 = floor(shift / spacing)
                # save all neighbours
                idx_list = np.asarray([idx_0 - 1, idx_0, idx_0 + 1, idx_0 + 2], dtype=np.int32) + ix

                # if idx_list[0] < 0: idx_list[0] += Npoints
                # if idx_list[3] > Npoints - 1: idx_list[3] -= Npoints
                # if idx_list[2] > Npoints - 1: idx_list[2] -= Npoints
                # subdiagonals needed if point is on other side of domain
                idx_subdiags_list = idx_list - Npoints
                # compute the distance to the index
                delta_idx = shift / spacing - idx_0
                # compute the 4 langrage basis elements
                lagrange_coefs = [lagrange(delta_idx, [-1, 0, 1, 2], j) for j in range(4)]
                # for the subdiagonals as well
                lagrange_coefs = lagrange_coefs #+ lagrange_coefs

                # band diagonals for the shift matrix

                offsets = [np.mod(idx, Npoints) for idx in idx_list]
                print(offsets)
                col += list(offsets)
                row += list(ix *np.ones_like(offsets))
                val += list(lagrange_coefs)


            val = np.asarray(val).flatten()
            Mat.append(sparse.coo_matrix((val,(row,col)), shape=(Nx,Nx)))

        return Mat

    # def compute_rotation_matrix(self, rotations, center, domain_length, spacing, Npoints):
    #     """

    #     Parameters
    #     ----------
    #     rotations : np.array of rotations size [#Snapshots]
    #                 This array should include the angle to the reference position
    #                 at every time point
    #     center : np.array of size 2
    #              (x,y) position of the center of rotation
    #     domain_length : np.array of size 2
                
    #     spacing : np.array of size 2
    #             lattice spacing
    #     Npoints : np.array Number of Gridpoints in x and y direction
            

    #     Returns
    #     -------
    #     Mat : Array of sparse matrix which rotate the field arround a center
    #          q(x',y', t) = M(omega(t))@ q(x,y,t) where q is a snapshot reshaped to a columnvector
    #          here (x')  = (cos(omega)* x - sin(omega)* y )
    #               (y')  = (sin(omega)* x + cos(omega)* y )
    #         Since the data only exists on discrete time points, we have to interpolate!
    #         The interpolation stencil ist defined in M(omega(t))
    #     """
    #     from numpy import floor
        
    #     Mat = []
    #     for shift in rotations:
            
    #         # we assume periodicity here
    #         if shift > domain_length:
    #             shift = shift - domain_length
    #         elif shift < 0:
    #             shift = shift + domain_length
        
    #         ''' interpolation scheme        lagrange_idx(x)= (x-x_{idx-1})/(x_idx - x_0)+
    #         -1      0   x    1       2                    ...+(x-x_{idx+2})/(x_idx - x_{idx+2})
    #          +      +   x    +       +
    #        idx_m1  idx_0    idx_1   idx_2
    #       =idx_0-1        =idx_0+1
    #        '''
           
    #         # shift is close to some discrete index:
    #         idx_0 = floor(shift/spacing)
    #         # save all neighbours
    #         idx_list = np.asarray([idx_0-1, idx_0, idx_0+1, idx_0+2],dtype=np.int32)
            
    #         if idx_list[0] < 0 : idx_list[0] += Npoints 
    #         if idx_list[3] > Npoints-1 : idx_list[3] -= Npoints
    #         # subdiagonals needed if point is on other side of domain
    #         idx_subdiags_list = idx_list - Npoints
    #         # compute the distance to the index
    #         delta_idx = shift/spacing - idx_0
    #         # compute the 4 langrage basis elements
    #         lagrange_coefs = [lagrange(delta_idx, [-1,0,1,2], j) for j in range(4)]
    #         # for the subdiagonals as well
    #         lagrange_coefs = lagrange_coefs +lagrange_coefs
            
    #         # band diagonals for the shift matrix
    #         offsets = np.concatenate([idx_list,idx_subdiags_list])
    #         diagonals = [np.ones(Npoints)*Lj  for Lj in lagrange_coefs]

   #         Mat.append(sparse.diags(diagonals,offsets,shape=[Npoints,Npoints]))
        
   #     return Mat

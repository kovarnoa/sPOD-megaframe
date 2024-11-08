#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
import sys
import os
import numpy as np
import functools
from generate_data import generate_data
sys.path.append("../lib")
from sPOD_algo import (
    shifted_POD,
    sPOD_Param,
)
from transforms import Transform
from platypus_sPOD_wrapper import *
sys.path.append("../thirdParty/Platypus")
import platypus

# ============================================================================ #

    

# ============================================================================ #
#                              CONSTANT DEFINITION                             #
# ============================================================================ #
#CASE = "crossing_waves"
#CASE = "three_cross_waves"        
#CASE = "sine_waves"
#CASE = "sine_waves_noise"
CASE = "multiple_ranks"
#CASE = "non_cross_mult_ranks"

Nx = 400        # number of grid points in x
Nt = Nx // 2    # number of time intervals
Niter = 1000    # number of sPOD iterations

nTasks = 1              # how much do we want to parallelize?
nEvaluations = 1000     # number of evaluations

#method = "ALM"
#method = "JFB"
#method = "ALM_megaframe"
method = "JFB_megaframe"

version = "V1"      # for outputs
outDir = './images/'
outFile =  outDir + 'results' + version + "_" + CASE + '.dat'

if not os.path.exists(outDir):
    os.makedirs(outDir)

# ============================================================================ #
#                                 Main Program                                 #
# ============================================================================ #
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

qmat = np.reshape(fields, [Nx, Nt])

nmodes = None
lambd0 = 1
myparams = sPOD_Param()
myparams.maxit = Niter
myparams.isError = False
if CASE == "sine_waves_noise":
    myparams.isError = True
myparams.isVerbose = False
myparams.total_variation_iterations = 0
myparams.eps = 1e-4

fixed = [qmat, transfos, method, myparams, outFile]        # putting all fixed inputs for sPOD into one structure


### DEFINE OPTIMIZED PARAMETERS
# var = optimized parameters for sPOD
#       JFB, no error matrix:   var = [lambda_s]
#       JFB, error matrix   :   var = [lambda_s, lambda_E]
#       ALM, no error matrix:   var = [lambda_s, mu/mu0]
#       ALM, error matrix   :   var = [lambda_s, mu/mu0, lambda_E]

var = ['lambda']
if (method == "ALM") or (method == "ALM_megaframe"):
    var.append('mu/mu0')
if myparams.isError:
    var.append('lambda_E')
nvar = len(var)

### DEFINE OBJECTIVES
# relative error, rank, (norm of error matrix)
obj = ['rel_err', 'rank']
if myparams.isError:
    obj.append('err_mat_norm')
nobj = len(obj)

head = ''
for objective in obj:
    head += objective
    head += '\t'
for variable in var:
    head += variable
    head += '\t'
print(head)

### DEFINE PROBLEM FOR PLATYPUS
problem             = platypus.Problem(nvar, nobj)
problem.types[:]    = platypus.Real(0, 10)          # optimized parameters go from 0 to 10 -- can be modified
#problem.types[0]    = platypus.Real(0.005, 0.5)
#print(problem.directions)
problem.function    = functools.partial(sPODOpt, fixed=fixed)

### RUN
with platypus.ProcessPoolEvaluator(nTasks) as evaluator:
    algorithm = platypus.NSGAII(problem, evaluator=evaluator) 
    algorithm.run(nEvaluations)

### OUTPUT
### put the results of the algorithm into array
nRes = len(algorithm.result)
out = []
for res in algorithm.result:
    if (res.objectives[0] < 1.0) and (res.objectives[1] > 0):       # only those that make sense -- rel err <1, rank > 0
        out.append(res)
nRes = len(out)
outMat = np.zeros((nRes, nobj + nvar))
for i, res in enumerate(out):
    outMat[i, :nobj] = out[i].objectives
    outMat[i, nobj:] = out[i].variables
outMat = outMat[outMat[:, 0].argsort()]     # sort by achieved relative error


# save  
fmt = ['%05.4e' for i in range(nobj+nvar)]
fmt[1] = '%d'   # second objective is rank -- integer

np.savetxt('./outMat_%s_%s.dat'%(method, version), outMat, header=head, fmt=fmt, delimiter='\t')


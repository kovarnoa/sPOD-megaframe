import sys
import os
import numpy as np
from generate_data import generate_data
sys.path.append("../lib")
from sPOD_algo import (
    shifted_POD,
    sPOD_Param,
)
from transforms import Transform
sys.path.append("../thirdParty/Platypus")

def sPODOpt(var, fixed):
    """
        Wrapper for sPOD used in optimization of proximal sPOD parameters via genetic algorithms
        
        fixed   ... all parameters not changed during optimization
        var     ... all parameters changed during optimization
                ... lambda_s (, mu/mu0, lambda_E)
    """
    qmat = fixed[0]
    transfos = fixed[1]
    method = fixed[2]
    myparams = fixed[3]
    
    varlamb = var[0]
    if (method == "ALM") or (method == "ALM_megaframe"):
        varmu  = var[1]
    else:
        varmu = 0
    if myparams.isError:
        varlamb_e = var[-1]     # it is second or third, but always last from var list
        myparams.lambda_E = varlamb_e

    mu0 = qmat.shape[0] * qmat.shape[1] / (4 * np.sum(np.abs(qmat)))      # article Krah(Marmin) 2024
    param_alm = mu0*varmu
    myparams.lambda_s = varlamb

    ret = shifted_POD(qmat, transfos, myparams, method, param_alm, nmodes=None)

    rel_err = ret.rel_err_hist[-1]
    rank = np.sum(ret.ranks)
    if myparams.isError:
        err_mat = np.linalg.norm(ret.error_matrix, ord=1)
    
    # print output
    if myparams.isError == False:
        out = '%06.5e\t%d\t'%(rel_err, rank)
    else:
        out = '%06.5e\t%d\t%06.5e\t'%(rel_err, rank, err_mat)    
    for v in var:
        out += '%06.5f\t'%(v)

    print(out)

    # return optimized stuff
    if myparams.isError == False:
        return [rel_err, rank]
    else:
        return [rel_err, rank, err_mat]

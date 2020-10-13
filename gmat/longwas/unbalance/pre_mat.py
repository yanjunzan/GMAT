import gc
import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import csr_matrix, isspmatrix, block_diag, lil_matrix, hstack, vstack
from scipy.sparse.linalg import inv
from .common import *


# inversion of covariance matrix
def pre_covi_mat(cov_dim, var_com):
    end = 0
    covi_mat = []  # covariance matrix list
    if var_com[-1] <= 0:
        return None
    for i in range(len(cov_dim)):
        start = end
        end = int(cov_dim[i] * (cov_dim[i] - 1) / 2) + cov_dim[i] + start
        covar = np.zeros((cov_dim[i], cov_dim[i]))
        covar[np.tril_indices(cov_dim[i])] = var_com[start:end]
        covar = np.add(covar, np.tril(covar, -1).T)
        try:
            linalg.cholesky(covar)
        except:
            return None
        covar = linalg.inv(covar)
        covi_mat.append(covar)
    
    return covi_mat


def pre_mat(y, xmat, eff_ind, zmat_con, cov_dim, var_com, covi_mat, \
            cmat_pure, kin, rhs_pure):
    # Coefficient Matrix
    cmat = (cmat_pure.multiply(1.0 / var_com[-1])).toarray()
    for i in range(len(cov_dim)):
        if isspmatrix(kin[i]):
            temp = sparse.kron(covi_mat[i], kin[i])
            temp = temp.toarray()
        else:
            temp = linalg.kron(covi_mat[i], kin[i])
        
        cmat[eff_ind[i + 1][0]:eff_ind[i + 1][-1], \
        eff_ind[i + 1][0]:eff_ind[i + 1][-1]] = \
            np.add(cmat[eff_ind[i + 1][0]:eff_ind[i + 1][-1], \
                   eff_ind[i + 1][0]:eff_ind[i + 1][-1]], temp)
    
    # right hand
    rhs_mat = np.divide(rhs_pure, var_com[-1])
    # all effects
    cmat = linalg.inv(cmat)
    eff = np.dot(cmat, rhs_mat)
    e = y - xmat.dot(eff[:eff_ind[0][1], :]) - zmat_con.dot(
        eff[eff_ind[0][1]:, :])
    
    return cmat, eff, e


def pre_sparse_mat(y, xmat, eff_ind, zmat_con, cov_dim, var_com, covi_mat, \
            cmat_pure, kin, rhs_pure):
    # Coefficient Matrix
    cmat = cmat_pure.multiply(1.0 / var_com[-1])
    '''
    for i in range(len(cov_dim)):
        temp = sparse.kron(covi_mat[i], kin[i])
        temp = temp.tolil()
        cmat[eff_ind[i + 1][0]:eff_ind[i + 1][-1], \
        eff_ind[i + 1][0]:eff_ind[i + 1][-1]] = \
            np.add(cmat[eff_ind[i + 1][0]:eff_ind[i + 1][-1], \
                   eff_ind[i + 1][0]:eff_ind[i + 1][-1]], temp)
    '''
    temp_vec = []
    for i in range(len(cov_dim)):
        temp = sparse.kron(covi_mat[i], kin[i])
        temp_vec.append(np.add(cmat[eff_ind[i + 1][0]:eff_ind[i + 1][-1], eff_ind[i + 1][0]:eff_ind[i + 1][-1]], temp))
    del temp
    gc.collect()
    cmat = vstack([cmat[:eff_ind[0][1], :],
     hstack([cmat[eff_ind[1][0]:eff_ind[1][-1], :eff_ind[1][0]], temp_vec[0],
             cmat[eff_ind[1][0]:eff_ind[1][-1], eff_ind[1][-1]:]]),
        hstack([cmat[eff_ind[2][0]:, :eff_ind[2][0]], temp_vec[1]])])
    del temp_vec
    gc.collect()
    # right hand
    rhs_mat = np.divide(rhs_pure, var_com[-1])
    
    # all effects
    # cmat = cmat.tocsc()
    # cmat = inv(cmat)
    # cmat = cmat.tocsr()
    # eff = cmat.dot(rhs_mat)
    # e = y - xmat.dot(eff[:eff_ind[0][1], :]) - zmat_con.dot(
    #     eff[eff_ind[0][1]:, :])
    
    return cmat, rhs_mat



def pre_mat_x(y, xmat, eff_ind, zmat_con, cov_dim, var_com, covi_mat, cmat_pure, kin, rhs_pure):
    # Coefficient Matrix
    cmat = (cmat_pure.multiply(1.0 / var_com[-1])).toarray()
    for i in range(len(cov_dim)):
        if isspmatrix(kin[i]):
            temp = sparse.kron(kin[i], covi_mat[i])
            temp = temp.toarray()
        else:
            temp = linalg.kron(kin[i], covi_mat[i])
        cmat[eff_ind[i + 1][0]:eff_ind[i + 1][-1], eff_ind[i + 1][0]:eff_ind[i + 1][-1]] = \
            np.add(cmat[eff_ind[i + 1][0]:eff_ind[i + 1][-1], eff_ind[i + 1][0]:eff_ind[i + 1][-1]], temp)
    
    # right hand
    rhs_mat = np.divide(rhs_pure, var_com[-1])
    
    # all effects
    cmat = linalg.inv(cmat)
    eff = np.dot(cmat, rhs_mat)
    e = y - xmat.dot(eff[:eff_ind[0][1], :]) - zmat_con.dot(
        eff[eff_ind[0][1]:, :])
    
    return cmat, eff, e


def pre_cov_mat_eigen(cov_dim, var_com):
    start = 0
    end = cov_dim * (cov_dim - 1) / 2 + cov_dim
    cov_add = np.zeros((cov_dim, cov_dim))
    cov_add[np.tril_indices_from(cov_add)] = var_com[start:end]
    cov_add = cov_add + np.tril(cov_add, k=-1).T
    try:
        linalg.cholesky(cov_add)
    except:
        return None
    start = end
    end = 2 * end
    cov_per = np.zeros((cov_dim, cov_dim))
    cov_per[np.tril_indices_from(cov_per)] = var_com[start:end]
    cov_per = cov_per + np.tril(cov_per, k=-1).T
    try:
        linalg.cholesky(cov_per)
    except:
        return None
    if var_com[-1] < 0:
        return None
    else:
        cov_res = np.array([[var_com[-1]]])
    return cov_add, cov_per, cov_res



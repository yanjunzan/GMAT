import gc
import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import csr_matrix, isspmatrix, block_diag, lil_matrix, hstack, vstack
from scipy.sparse.linalg import inv


def pre_covar_inv(covar_dim, var_com):
    # inversion of covariance matrix
    end = 0
    covar_inv = []  # covariance matrix list
    if var_com[-1] <= 0:
        return None
    for i in range(len(covar_dim)):
        start = end
        ind_sel = np.tril_indices(covar_dim[i])
        end = len(ind_sel[0]) + start
        covar = np.zeros((covar_dim[i], covar_dim[i]))
        covar[ind_sel] = var_com[start:end]
        covar = np.add(covar, np.tril(covar, -1).T)
        try:
            linalg.cholesky(covar)
        except Exception as e:
            # print(e)
            return None
        covar = linalg.inv(covar)
        covar_inv.append(covar)
    
    return covar_inv


def pre_coef_mat(y, xmat, eff_ind, zmat_con, covar_dim, var_com, covar_inv, coef_mat_pure, rmat_inv_lst, rhs_pure):
    # Coefficient Matrix
    coef_mat = (coef_mat_pure.multiply(1.0 / var_com[-1])).toarray()
    for i in range(len(covar_dim)):
        if isspmatrix(rmat_inv_lst[i]):
            temp = sparse.kron(covar_inv[i], rmat_inv_lst[i])
            temp = temp.toarray()
        else:
            temp = linalg.kron(covar_inv[i], rmat_inv_lst[i])
        
        coef_mat[eff_ind[i + 1][0]:eff_ind[i + 1][-1], eff_ind[i + 1][0]:eff_ind[i + 1][-1]] = \
            np.add(coef_mat[eff_ind[i + 1][0]:eff_ind[i + 1][-1], eff_ind[i + 1][0]:eff_ind[i + 1][-1]], temp)
    
    # right hand
    rhs_mat = np.divide(rhs_pure, var_com[-1])
    
    # all effects
    coef_mat = linalg.inv(coef_mat)
    eff = np.dot(coef_mat, rhs_mat)
    
    # errors
    e = y - xmat.dot(eff[:eff_ind[0][1], :]) - zmat_con.dot(
        eff[eff_ind[0][1]:, :])
    
    return coef_mat, eff, e


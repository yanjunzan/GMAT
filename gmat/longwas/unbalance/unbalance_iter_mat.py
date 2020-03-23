import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import isspmatrix


def unbalance_pre_fd_mat(coef_mat_inv, kin, covar_inv, eff, eff_ind, e, covar_dim, zmat_concat_lst, wmat, num_record,
                         var_com):
    fd_mat = []
    for i in range(len(covar_dim)):
        qnum = zmat_concat_lst[i].shape[1] / covar_dim[i]
        tmat = np.array([0.0] * covar_dim[i] * covar_dim[i]).reshape(covar_dim[i], covar_dim[i])
        eff_lst = []
        if not isspmatrix(kin[i]):
            for j in range(covar_dim[i]):
                for k in range(j + 1):
                    tmat[j, k] = np.sum(np.multiply(kin[i], coef_mat_inv[eff_ind[i + 1][j]:eff_ind[i + 1][j + 1],
                                                            eff_ind[i + 1][k]:eff_ind[i + 1][k + 1]]))
                    tmat[k, j] = tmat[j, k]
                eff_lst.append(eff[eff_ind[i + 1][j]:eff_ind[i + 1][j + 1], :])
            eff_mat = np.dot(np.concatenate(eff_lst, axis=1), covar_inv[i])
            temp = qnum * covar_inv[i] - np.dot(np.dot(covar_inv[i], tmat), covar_inv[i]) \
                   - np.dot(np.dot(eff_mat.T, kin[i]), eff_mat)
        else:
            for j in range(covar_dim[i]):
                for k in range(j + 1):
                    tmat[j, k] = np.sum(kin[i].multiply(coef_mat_inv[eff_ind[i + 1][j]:eff_ind[i + 1][j + 1],
                                                        eff_ind[i + 1][k]:eff_ind[i + 1][k + 1]]))
                    tmat[k, j] = tmat[j, k]
                eff_lst.append(eff[eff_ind[i + 1][j]:eff_ind[i + 1][j + 1], :])
            eff_mat = np.dot(np.concatenate(eff_lst, axis=1), covar_inv[i])
            temp = qnum * covar_inv[i] - np.dot(np.dot(covar_inv[i], tmat), covar_inv[i]) \
                   - np.dot(eff_mat.T, kin[i].dot(eff_mat))
        
        temp[np.tril_indices(covar_dim[i], k=-1)] = 2.0 * temp[np.tril_indices(covar_dim[i], k=-1)]
        fd_mat.extend(list(-0.5 * temp[np.tril_indices(covar_dim[i], k=0)]))
    
    temp = -0.5 * (num_record / var_com[-1] - np.sum((wmat.T.dot(wmat)).multiply(coef_mat_inv)) \
                   / (var_com[-1] * var_com[-1]) - np.sum(np.dot(e.T, e) / (var_com[-1] * var_com[-1])))
    
    fd_mat.append(temp)
    # print "FD matrix:", fd_mat
    fd_mat = np.array(fd_mat)
    return fd_mat


def unbalance_pre_ai_mat(coef_mat_inv, covar_inv, eff, eff_ind, e, covar_dim, zmat_concat_lst, wmat, var_com):
    wv = []   # working variable
    for i in range(len(covar_dim)):
        dial = sparse.eye(zmat_concat_lst[i].shape[1] / covar_dim[i], dtype=np.float64)
        for j in range(covar_dim[i]):
            for k in range(j + 1):
                var_fd = np.zeros((covar_dim[i], covar_dim[i]))  # Notice
                var_fd[j, k] = var_fd[k, j] = 1.0
                temp = sparse.kron(np.dot(var_fd, covar_inv[i]), dial)
                temp = temp.dot(eff[eff_ind[i + 1][0]:eff_ind[i + 1][-1], :])
                temp = zmat_concat_lst[i].dot(temp)
                wv.append(temp)
    wv.append(e / var_com[-1])
    qmat = np.concatenate(wv, axis=1)
    
    # AI matrix
    qrq = np.divide(np.dot(qmat.T, qmat), var_com[-1])
    left = np.divide(wmat.T.dot(qmat), var_com[-1])
    eff_qmat = np.dot(coef_mat_inv, left)
    ai_mat = np.subtract(qrq, np.dot(left.T, eff_qmat))
    ai_mat = 0.5 * ai_mat
    try:
        linalg.cholesky(ai_mat)
    except Exception as e:
        print(e)
        ai_mat = False
    return ai_mat


def unbalance_pre_em_mat(covar_dim, zmat_concat_lst, num_record, var_com):
    num_var_com = len(var_com)
    em_mat = np.array([0.0] * num_var_com * num_var_com).reshape(num_var_com, num_var_com)
    b = 0
    cov_mat = []
    for i in range(len(covar_dim)):
        ind = np.tril_indices(covar_dim[i])
        temp = np.array([0.0] * len(ind[0]) * len(ind[0])).reshape(len(ind[0]), len(ind[0]))
        q = zmat_concat_lst[i].shape[1] / covar_dim[i]
        a = b
        b = b + len(ind[0])
        covar = np.zeros((covar_dim[i], covar_dim[i]))
        covar[ind] = var_com[a:b]
        covar = np.add(covar, np.tril(covar, -1).T)
        cov_mat.append(covar)
        for j in range(len(ind[0])):
            for k in range(j + 1):
                temp[j, k] = (cov_mat[i][ind[0][j], ind[0][k]] * cov_mat[i][ind[1][j], ind[1][k]] + \
                              cov_mat[i][ind[0][j], ind[1][k]] * cov_mat[i][ind[1][j], ind[0][k]]) / (2.0 * q)
        em_mat[a:b, a:b] = temp.copy()
    em_mat[-1, -1] = (var_com[-1] * var_com[-1]) / num_record
    em_mat = 2.0 * em_mat
    
    em_mat += np.tril(em_mat, k=-1).T
    return linalg.inv(em_mat)

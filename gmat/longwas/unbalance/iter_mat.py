import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import isspmatrix
from .common import *


def pre_fd_mat(cmati, kin, covi_mat, eff, eff_ind, e, cov_dim, zmat_con_lst, wmat, num_record, var_com):
    fd_mat = []
    for i in range(len(cov_dim)):
        for j in range(cov_dim[i]):
            for k in range(j + 1):
                var_fd = np.zeros((cov_dim[i], cov_dim[i]))
                var_fd[j, k] = var_fd[k, j] = 1.0
                fd1 = np.sum(np.multiply(var_fd, covi_mat[i])) * zmat_con_lst[i].shape[1] / cov_dim[i]
                if isspmatrix(kin[i]):
                    temp = np.dot(covi_mat[i], np.dot(var_fd, covi_mat[i]))
                    temp = sparse.kron(temp, kin[i])
                    fd2 = np.sum(temp.multiply(cmati[eff_ind[i + 1][0]:eff_ind[i + 1][-1], \
                                               eff_ind[i + 1][0]:eff_ind[i + 1][-1]]))
                    fd3 = np.sum(np.dot(eff[eff_ind[i + 1][0]:eff_ind[i + 1][-1], :].T, \
                                        temp.dot(eff[eff_ind[i + 1][0]:eff_ind[i + 1][-1], :])))
                else:
                    temp = np.dot(covi_mat[i], np.dot(var_fd, covi_mat[i]))
                    temp = linalg.kron(temp, kin[i])
                    fd2 = np.sum(np.multiply(temp, cmati[eff_ind[i + 1][0]:eff_ind[i + 1][-1], \
                                                   eff_ind[i + 1][0]:eff_ind[i + 1][-1]]))
                    fd3 = np.sum(np.dot(eff[eff_ind[i + 1][0]:eff_ind[i + 1][-1], :].T, \
                                        np.dot(temp, eff[eff_ind[i + 1][0]:eff_ind[i + 1][-1], :])))
                fd_mat.append(-0.5 * (fd1 - fd2 - fd3))
    
    temp = -0.5 * (num_record / var_com[-1] - np.sum((wmat.T.dot(wmat)).multiply(cmati)) \
                   / (var_com[-1] * var_com[-1]) - np.sum(np.dot(e.T, e) / (var_com[-1] * var_com[-1])))
    fd_mat.append(temp)
    # print "FD matrix:", fd_mat
    fd_mat = np.array(fd_mat)
    return fd_mat


def pre_fd_mat_x(cmati, kin, covi_mat, eff, eff_ind, e, cov_dim, zmat_con_lst, wmat, num_record, var_com):
    fd_mat = []
    for i in range(len(cov_dim)):
        qnum = zmat_con_lst[i].shape[1] / cov_dim[i]
        tmat = np.array([0.0] * cov_dim[i] * cov_dim[i]).reshape(cov_dim[i], cov_dim[i])
        eff_lst = []
        if not isspmatrix(kin[i]):
            for j in range(cov_dim[i]):
                for k in range(j + 1):
                    tmat[j, k] = np.sum(np.multiply(kin[i], cmati[eff_ind[i + 1][j]:eff_ind[i + 1][j + 1],
                                                            eff_ind[i + 1][k]:eff_ind[i + 1][k + 1]]))
                    tmat[k, j] = tmat[j, k]
                eff_lst.append(eff[eff_ind[i + 1][j]:eff_ind[i + 1][j + 1], :])
            eff_mat = np.dot(np.concatenate(eff_lst, axis=1), covi_mat[i])
            temp = qnum * covi_mat[i] - np.dot(np.dot(covi_mat[i], tmat), covi_mat[i]) \
                   - np.dot(np.dot(eff_mat.T, kin[i]), eff_mat)
        else:
            for j in range(cov_dim[i]):
                for k in range(j + 1):
                    tmat[j, k] = np.sum(kin[i].multiply(cmati[eff_ind[i + 1][j]:eff_ind[i + 1][j + 1],
                                                        eff_ind[i + 1][k]:eff_ind[i + 1][k + 1]]))
                    tmat[k, j] = tmat[j, k]
                eff_lst.append(eff[eff_ind[i + 1][j]:eff_ind[i + 1][j + 1], :])
            eff_mat = np.dot(np.concatenate(eff_lst, axis=1), covi_mat[i])
            temp = qnum * covi_mat[i] - np.dot(np.dot(covi_mat[i], tmat), covi_mat[i]) \
                   - np.dot(eff_mat.T, kin[i].dot(eff_mat))
        
        temp[np.tril_indices(cov_dim[i], k=-1)] = 2.0 * temp[np.tril_indices(cov_dim[i], k=-1)]
        fd_mat.extend(list(-0.5 * temp[np.tril_indices(cov_dim[i], k=0)]))
    
    temp = -0.5 * (num_record / var_com[-1] - np.sum((wmat.T.dot(wmat)).multiply(cmati)) \
                   / (var_com[-1] * var_com[-1]) - np.sum(np.dot(e.T, e) / (var_com[-1] * var_com[-1])))
    
    fd_mat.append(temp)
    # print "FD matrix:", fd_mat
    fd_mat = np.array(fd_mat)
    return fd_mat


def pre_fd_sparse_mat(cmati, kin, covi_mat, eff, eff_ind, e, cov_dim, zmat_con_lst, wmat, num_record, var_com):
    fd_mat = []
    for i in range(len(cov_dim)):
        qnum = zmat_con_lst[i].shape[1] / cov_dim[i]
        tmat = np.array([0.0] * cov_dim[i] * cov_dim[i]).reshape(cov_dim[i], cov_dim[i])
        eff_lst = []
        for j in range(cov_dim[i]):
            for k in range(j + 1):
                tmat[j, k] = np.sum(kin[i].multiply(cmati[eff_ind[i + 1][j]:eff_ind[i + 1][j + 1],
                                                       eff_ind[i + 1][k]:eff_ind[i + 1][k + 1]]))
                tmat[k, j] = tmat[j, k]
            eff_lst.append(eff[eff_ind[i + 1][j]:eff_ind[i + 1][j + 1], :])
        eff_mat = np.dot(np.concatenate(eff_lst, axis=1), covi_mat[i])
        temp = qnum * covi_mat[i] - np.dot(np.dot(covi_mat[i], tmat), covi_mat[i]) \
                   -np.dot(eff_mat.T, kin[i].dot(eff_mat))
        temp[np.tril_indices(cov_dim[i], k=-1)] = 2.0 * temp[np.tril_indices(cov_dim[i], k=-1)]
        fd_mat.extend(list(-0.5 * temp[np.tril_indices(cov_dim[i], k=0)]))
    
    temp = -0.5 * (num_record / var_com[-1] - np.sum((wmat.T.dot(wmat)).multiply(cmati)) / (var_com[-1] * var_com[
        -1]) - np.sum(np.dot(e.T, e) / (var_com[-1] * var_com[-1])))
    
    fd_mat.append(temp)
    fd_mat = np.array(fd_mat)
    return fd_mat


def pre_ai_mat(cmati, covi_mat, eff, eff_ind, e, cov_dim, zmat_con_lst, wmat, var_com):
    ###working variable
    wv = []
    for i in range(len(cov_dim)):
        dial = sparse.eye(zmat_con_lst[i].shape[1] / cov_dim[i], dtype=np.float64)
        for j in range(cov_dim[i]):
            for k in range(j + 1):
                var_fd = np.zeros((cov_dim[i], cov_dim[i]))  # Notice
                var_fd[j, k] = var_fd[k, j] = 1.0
                temp = sparse.kron(np.dot(var_fd, covi_mat[i]), dial)
                temp = temp.dot(eff[eff_ind[i + 1][0]:eff_ind[i + 1][-1], :])
                temp = zmat_con_lst[i].dot(temp)
                wv.append(temp)
    wv.append(e / var_com[-1])
    qmat = np.concatenate(wv, axis=1)
    # print "Working variables: ", Q[:,:5]
    
    ###AI matrix
    qrq = np.divide(np.dot(qmat.T, qmat), var_com[-1])
    left = np.divide(wmat.T.dot(qmat), var_com[-1])
    eff_qmat = np.dot(cmati, left)
    ai_mat = np.subtract(qrq, np.dot(left.T, eff_qmat))
    ai_mat = 0.5 * ai_mat
    try:
        linalg.cholesky(ai_mat)
    except:
        ai_mat = False
    return ai_mat


def pre_ai_sparse_mat(cmati, covi_mat, eff, eff_ind, e, cov_dim, zmat_con_lst, wmat, var_com):
    ###working variable
    wv = []
    for i in range(len(cov_dim)):
        dial = sparse.eye(zmat_con_lst[i].shape[1] / cov_dim[i], dtype=np.float64)
        for j in range(cov_dim[i]):
            for k in range(j + 1):
                var_fd = np.zeros((cov_dim[i], cov_dim[i]))  # Notice
                var_fd[j, k] = var_fd[k, j] = 1.0
                temp = sparse.kron(np.dot(var_fd, covi_mat[i]), dial)
                temp = temp.dot(eff[eff_ind[i + 1][0]:eff_ind[i + 1][-1], :])
                temp = zmat_con_lst[i].dot(temp)
                wv.append(temp)
    wv.append(e / var_com[-1])
    qmat = np.concatenate(wv, axis=1)
    # print "Working variables: ", Q[:,:5]
    
    ###AI matrix
    qrq = np.divide(np.dot(qmat.T, qmat), var_com[-1])
    left = np.divide(wmat.T.dot(qmat), var_com[-1])
    eff_qmat = cmati.dot(left)
    ai_mat = np.subtract(qrq, np.dot(left.T, eff_qmat))
    ai_mat = 0.5 * ai_mat
    try:
        linalg.cholesky(ai_mat)
    except:
        ai_mat = False
    return ai_mat


def pre_em_mat(cov_dim, zmat_con_lst, num_record, var_com):
    num_var_com = len(var_com)
    em_mat = np.array([0.0] * num_var_com * num_var_com).reshape(num_var_com, num_var_com)
    b = 0
    cov_mat = []
    for i in range(len(cov_dim)):
        ind = np.tril_indices(cov_dim[i])
        temp = np.array([0.0] * len(ind[0]) * len(ind[0])).reshape(len(ind[0]), len(ind[0]))
        q = zmat_con_lst[i].shape[1] / cov_dim[i]
        a = b
        b = b + len(ind[0])
        covar = np.zeros((cov_dim[i], cov_dim[i]))
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


def pre_fdai_mat_eigen(y, xmat, leg_tp, kin_eigen_val, cov_add, cov_per, cov_res, var_com, var_ind, kin_dct):
    vinv = np.multiply(kin_eigen_val.reshape(len(kin_eigen_val), 1, 1), tri_matT(leg_tp, cov_add)) + \
           tri_matT(leg_tp, cov_per) + np.diag([np.sum(cov_res)] * leg_tp.shape[0])
    vinv = np.linalg.inv(vinv)
    
    xv = np.matmul(xmat.transpose(0, 2, 1), vinv)
    xvx = np.matmul(xv, xmat)
    xvx = reduce(np.add, xvx)
    xvx = np.linalg.inv(xvx)
    
    # xvy
    yv = np.matmul(y.transpose(0, 2, 1), vinv)
    yvx = np.matmul(yv, xmat)
    yvx = reduce(np.add, yvx)
    
    fd_mat = np.array([0.0] * len(var_com))
    ai_mat = np.array([0.0] * len(var_com) * len(var_com)).reshape(len(var_com), len(var_com))
    
    for m in range(var_ind.shape[0]):
        
        onei = np.array([0.0] * leg_tp.shape[1]).reshape(leg_tp.shape[1], 1)
        onej = onei.copy()
        onei[var_ind[m, 1], 0] = 1.0
        onej[var_ind[m, 2], 0] = 1.0
        one = np.dot(onei, onej.T)
        one = one + one.T - np.diag(np.diag(one))
        tit = reduce(np.dot, [leg_tp, one, leg_tp.T])
        if m == var_ind.shape[0] - 1:
            tit = np.eye(leg_tp.shape[0])
        
        # tr(vk)
        trvk = np.matmul(vinv, tit)
        trvk = np.multiply(kin_dct[var_ind[m, 0]], trvk)
        trvk = np.trace(reduce(np.add, trvk))
        
        # tr(pk)
        xvkvx = reduce(np.matmul, [xv, tit, xv.transpose(0, 2, 1)])
        xvkvx = np.multiply(kin_dct[var_ind[m, 0]], xvkvx)
        xvkvx = reduce(np.add, xvkvx)
        trpk = np.sum(np.multiply(xvx, xvkvx))
        # ypkpy
        yvkvy = reduce(np.matmul, [yv, tit, yv.transpose(0, 2, 1)])
        yvkvy = np.multiply(kin_dct[var_ind[m, 0]], yvkvy)
        yvkvy = reduce(np.add, yvkvy)
        yvkvx = reduce(np.matmul, [yv, tit, xv.transpose(0, 2, 1)])
        yvkvx = np.multiply(kin_dct[var_ind[m, 0]], yvkvx)
        yvkvx = reduce(np.add, yvkvx)
        ypkpy2 = reduce(np.dot, [yvx, xvx, yvkvx.T])
        ypkpy3 = np.dot(yvx, xvx)
        ypkpy3 = reduce(np.dot, [ypkpy3, xvkvx, ypkpy3.T])
        ypkpy = np.sum(yvkvy) - 2 * np.sum(ypkpy2) + np.sum(ypkpy3)
        fd_mat[m] = -0.5 * (trvk - trpk) + 0.5 * ypkpy
        
        for n in range(m + 1):
            onei2 = np.array([0.0] * leg_tp.shape[1]).reshape(leg_tp.shape[1], 1)
            onej2 = onei2.copy()
            onei2[var_ind[n, 1], 0] = 1.0
            onej2[var_ind[n, 2], 0] = 1.0
            one2 = np.dot(onei2, onej2.T)
            one2 = one2 + one2.T - np.diag(np.diag(one2))
            tit2 = reduce(np.dot, [leg_tp, one2, leg_tp.T])
            if n == var_ind.shape[0] - 1:
                tit2 = np.eye(leg_tp.shape[0])
            yvkvkvy = reduce(np.matmul, [yv, tit, vinv, tit2, yv.transpose(0, 2, 1)])
            yvkvkvy = reduce(np.multiply, [kin_dct[var_ind[m, 0]], kin_dct[var_ind[n, 0]], yvkvkvy])
            val1 = reduce(np.add, yvkvkvy)
            
            yvkvkvx = reduce(np.matmul, [yv, tit, vinv, tit2, xv.transpose(0, 2, 1)])
            yvkvkvx = reduce(np.multiply, [kin_dct[var_ind[m, 0]], kin_dct[var_ind[n, 0]], yvkvkvx])
            yvkvkvx = reduce(np.add, yvkvkvx)
            val2 = reduce(np.dot, [yvkvkvx, xvx, yvx.T])
            
            yvkvx2 = reduce(np.matmul, [yv, tit2, xv.transpose(0, 2, 1)])
            yvkvx2 = np.multiply(kin_dct[var_ind[n, 0]], yvkvx2)
            yvkvx2 = reduce(np.add, yvkvx2)
            val3 = reduce(np.dot, [yvkvx, xvx, yvkvx2.T])
            
            xvkvx2 = reduce(np.matmul, [xv, tit2, xv.transpose(0, 2, 1)])
            xvkvx2 = np.multiply(kin_dct[var_ind[n, 0]], xvkvx2)
            xvkvx2 = reduce(np.add, xvkvx2)
            val4 = reduce(np.dot, [yvkvx, xvx, xvkvx2, xvx, yvx.T])
            
            yvkvkvx2 = reduce(np.matmul, [yv, tit2, vinv, tit, xv.transpose(0, 2, 1)])
            yvkvkvx2 = reduce(np.multiply, [kin_dct[var_ind[m, 0]], kin_dct[var_ind[n, 0]], yvkvkvx2])
            yvkvkvx2 = reduce(np.add, yvkvkvx2)
            val5 = reduce(np.dot, [yvx, xvx, yvkvkvx2.T])
            
            xvkvkvx2 = reduce(np.matmul, [xv, tit, vinv, tit2, xv.transpose(0, 2, 1)])
            xvkvkvx2 = reduce(np.multiply, [kin_dct[var_ind[m, 0]], kin_dct[var_ind[n, 0]], xvkvkvx2])
            xvkvkvx2 = reduce(np.add, xvkvkvx2)
            val6 = reduce(np.dot, [yvx, xvx, xvkvkvx2, xvx, yvx.T])
            
            val7 = reduce(np.dot, [yvx, xvx, xvkvx, xvx, yvkvx2.T])
            val8 = reduce(np.dot, [yvx, xvx, xvkvx, xvx, xvkvx2, xvx, yvx.T])
            temp = val1 - val2 - val3 + val4 - val5 + val6 + val7 - val8
            ai_mat[m, n] = 0.5 * np.sum(temp)
    ai_mat = ai_mat + np.tril(ai_mat, k=-1).T
    
    return fd_mat, ai_mat


def pre_em_mat_eigen(cov_dim, cov_add, cov_per, ran_df, var_com, num_record):
    num_var_com = len(var_com)
    em_mat = np.array([0.0] * num_var_com * num_var_com).reshape(num_var_com, num_var_com)
    
    ind = np.tril_indices(cov_dim)
    temp = np.array([0.0] * len(ind[0]) * len(ind[0])).reshape(len(ind[0]), len(ind[0]))
    for j in range(len(ind[0])):
        for k in range(j + 1):
            temp[k, j] = temp[j, k] = (cov_add[ind[0][j], ind[0][k]] * cov_add[ind[1][j], ind[1][k]] + \
                                       cov_add[ind[0][j], ind[1][k]] * cov_add[ind[1][j], ind[0][k]]) / (2.0 * ran_df)
    temp = linalg.inv(temp)
    em_mat[0:len(ind[0]), 0:len(ind[0])] = temp.copy()
    
    temp = np.array([0.0] * len(ind[0]) * len(ind[0])).reshape(len(ind[0]), len(ind[0]))
    for j in range(len(ind[0])):
        for k in range(j + 1):
            temp[k, j] = temp[j, k] = (cov_per[ind[0][j], ind[0][k]] * cov_per[ind[1][j], ind[1][k]] + \
                                       cov_per[ind[0][j], ind[1][k]] * cov_per[ind[1][j], ind[0][k]]) / (2.0 * ran_df)
    temp = linalg.inv(temp)
    em_mat[len(ind[0]):2 * len(ind[0]), len(ind[0]):2 * len(ind[0])] = temp.copy()
    
    em_mat[-1, -1] = num_record / (var_com[-1] * var_com[-1])
    
    em_mat = em_mat / 2.0
    return em_mat


def pre_fdai_mat_eigen_x(y, xmat, leg_tp, kin_eigen_val, cov_add, cov_per, cov_res, var_com, var_ind, kin_dct):
    vinv = np.multiply(kin_eigen_val.reshape(len(kin_eigen_val), 1, 1), tri_matT(leg_tp, cov_add)) + \
           tri_matT(leg_tp, cov_per) + np.diag([np.sum(cov_res)] * leg_tp.shape[0])
    vinv = np.linalg.inv(vinv)
    
    xvx = reduce(np.matmul, [xmat.transpose(0, 2, 1), vinv, xmat])
    xvx = reduce(np.add, xvx)
    xvx = np.linalg.inv(xvx)
    
    # xvy
    yv = np.matmul(y.transpose(0, 2, 1), vinv)
    yvx = np.matmul(yv, xmat)
    yvx = reduce(np.add, yvx)
    
    yvvx = reduce(np.matmul, [yv, vinv, xmat])
    yvvx = reduce(np.add, yvvx)
    
    xvvx = reduce(np.matmul, [xmat.transpose(0, 2, 1), vinv, vinv, xmat])
    xvvx = reduce(np.add, xvvx)
    
    fd_mat = np.array([0.0] * len(var_com))
    ai_mat = np.array([0.0] * len(var_com) * len(var_com)).reshape(len(var_com), len(var_com))
    
    for m in range(var_ind.shape[0] - 1):
        onei = np.array([0.0] * leg_tp.shape[1]).reshape(leg_tp.shape[1], 1)
        onej = onei.copy()
        onei[var_ind[m, 1], 0] = 1.0
        onej[var_ind[m, 2], 0] = 1.0
        one = np.dot(onei, onej.T)
        one = one + one.T - np.diag(np.diag(one))
        tit = reduce(np.dot, [leg_tp, one, leg_tp.T])
        
        # tr(vk)
        trvk = reduce(np.matmul, [onej.T, leg_tp.T, vinv, leg_tp, onei])
        trvk = np.multiply(kin_dct[var_ind[m, 0]], trvk)
        trvk = np.sum(reduce(np.add, trvk))
        
        if var_ind[m, 1] != var_ind[m, 2]:
            trvk = 2 * trvk
        
        # tr(pk)
        temp2 = reduce(np.matmul, [onej.T, leg_tp.T, vinv, xmat])
        temp1 = reduce(np.matmul, [onei.T, leg_tp.T, vinv, xmat])
        xvkvx = np.matmul(temp1.transpose(0, 2, 1), temp2)
        xvkvx = np.multiply(kin_dct[var_ind[m, 0]], xvkvx)
        xvkvx = reduce(np.add, xvkvx)
        if var_ind[m, 1] != var_ind[m, 2]:
            xvkvx = xvkvx + xvkvx.T
        trpk = np.sum(np.multiply(xvx, xvkvx))
        
        # ypkpy
        yvkvy = reduce(np.matmul, [yv, tit, yv.transpose(0, 2, 1)])
        yvkvy = np.multiply(kin_dct[var_ind[m, 0]], yvkvy)
        yvkvy = reduce(np.add, yvkvy)
        yvkvx = reduce(np.matmul, [yv, tit, vinv, xmat])
        yvkvx = np.multiply(kin_dct[var_ind[m, 0]], yvkvx)
        yvkvx = reduce(np.add, yvkvx)
        ypkpy2 = reduce(np.dot, [yvx, xvx, yvkvx.T])
        ypkpy3 = np.dot(yvx, xvx)
        ypkpy3 = reduce(np.dot, [ypkpy3, xvkvx, ypkpy3.T])
        ypkpy = np.sum(yvkvy) - 2 * np.sum(ypkpy2) + np.sum(ypkpy3)
        fd_mat[m] = -0.5 * (trvk - trpk) + 0.5 * ypkpy
        
        for n in range(m + 1):
            onei2 = np.array([0.0] * leg_tp.shape[1]).reshape(leg_tp.shape[1], 1)
            onej2 = onei2.copy()
            onei2[var_ind[n, 1], 0] = 1.0
            onej2[var_ind[n, 2], 0] = 1.0
            one2 = np.dot(onei2, onej2.T)
            one2 = one2 + one2.T - np.diag(np.diag(one2))
            tit2 = reduce(np.dot, [leg_tp, one2, leg_tp.T])
            yvkvkvy = reduce(np.matmul, [yv, tit, vinv, tit2, yv.transpose(0, 2, 1)])
            yvkvkvy = reduce(np.multiply, [kin_dct[var_ind[m, 0]], kin_dct[var_ind[n, 0]], yvkvkvy])
            val1 = reduce(np.add, yvkvkvy)
            
            yvkvkvx = reduce(np.matmul, [yv, tit, vinv, tit2, vinv, xmat])
            yvkvkvx = reduce(np.multiply, [kin_dct[var_ind[m, 0]], kin_dct[var_ind[n, 0]], yvkvkvx])
            yvkvkvx = reduce(np.add, yvkvkvx)
            val2 = reduce(np.dot, [yvkvkvx, xvx, yvx.T])
            
            yvkvx2 = reduce(np.matmul, [yv, tit2, vinv, xmat])
            yvkvx2 = np.multiply(kin_dct[var_ind[n, 0]], yvkvx2)
            yvkvx2 = reduce(np.add, yvkvx2)
            val3 = reduce(np.dot, [yvkvx, xvx, yvkvx2.T])
            
            temp2 = reduce(np.matmul, [onej2.T, leg_tp.T, vinv, xmat])
            temp1 = reduce(np.matmul, [onei2.T, leg_tp.T, vinv, xmat])
            xvkvx2 = np.matmul(temp1.transpose(0, 2, 1), temp2)
            xvkvx2 = np.multiply(kin_dct[var_ind[n, 0]], xvkvx2)
            xvkvx2 = reduce(np.add, xvkvx2)
            if var_ind[n, 1] != var_ind[n, 2]:
                xvkvx2 = xvkvx2 + xvkvx2.T
            val4 = reduce(np.dot, [yvkvx, xvx, xvkvx2, xvx, yvx.T])
            
            yvkvkvx2 = reduce(np.matmul, [yv, tit2, vinv, tit, vinv, xmat])
            yvkvkvx2 = reduce(np.multiply, [kin_dct[var_ind[m, 0]], kin_dct[var_ind[n, 0]], yvkvkvx2])
            yvkvkvx2 = reduce(np.add, yvkvkvx2)
            val5 = reduce(np.dot, [yvx, xvx, yvkvkvx2.T])
            
            xvkvkvx2 = cal_xvkvkvx(xmat, vinv, onei, onej, onei2, onej2, leg_tp, kin_dct, var_ind, m, n)
            if var_ind[n, 1] != var_ind[n, 2] and var_ind[m, 1] != var_ind[m, 2]:
                xvkvkvx2 = xvkvkvx2 + \
                           cal_xvkvkvx(xmat, vinv, onej, onei, onei2, onej2, leg_tp, kin_dct, var_ind, m, n) \
                           + cal_xvkvkvx(xmat, vinv, onei, onej, onej2, onei2, leg_tp, kin_dct, var_ind, m, n) \
                           + cal_xvkvkvx(xmat, vinv, onej, onei, onej2, onei2, leg_tp, kin_dct, var_ind, m, n)
            elif var_ind[m, 1] != var_ind[m, 2]:
                xvkvkvx2 = xvkvkvx2 + cal_xvkvkvx(xmat, vinv, onej, onei, onei2, onej2, leg_tp, kin_dct, var_ind, m, n)
            elif var_ind[n, 1] != var_ind[n, 2]:
                xvkvkvx2 = xvkvkvx2 + cal_xvkvkvx(xmat, vinv, onei, onej, onej2, onei2, leg_tp, kin_dct, var_ind, m, n)
            val6 = reduce(np.dot, [yvx, xvx, xvkvkvx2, xvx, yvx.T])
            
            val7 = reduce(np.dot, [yvx, xvx, xvkvx, xvx, yvkvx2.T])
            val8 = reduce(np.dot, [yvx, xvx, xvkvx, xvx, xvkvx2, xvx, yvx.T])
            temp = val1 - val2 - val3 + val4 - val5 + val6 + val7 - val8
            ai_mat[m, n] = 0.5 * np.sum(temp)
        
        yvkvkvy = reduce(np.matmul, [yv, tit, vinv, yv.transpose(0, 2, 1)])
        yvkvkvy = reduce(np.multiply, [kin_dct[var_ind[m, 0]], yvkvkvy])
        val1 = reduce(np.add, yvkvkvy)
        
        yvkvkvx = reduce(np.matmul, [yv, tit, vinv, vinv, xmat])
        yvkvkvx = reduce(np.multiply, [kin_dct[var_ind[m, 0]], yvkvkvx])
        yvkvkvx = reduce(np.add, yvkvkvx)
        val2 = reduce(np.dot, [yvkvkvx, xvx, yvx.T])
        
        val3 = reduce(np.dot, [yvkvx, xvx, yvvx.T])
        
        val4 = reduce(np.dot, [yvkvx, xvx, xvvx, xvx, yvx.T])
        
        yvkvkvx2 = reduce(np.matmul, [yv, vinv, tit, vinv, xmat])
        yvkvkvx2 = reduce(np.multiply, [kin_dct[var_ind[m, 0]], yvkvkvx2])
        yvkvkvx2 = reduce(np.add, yvkvkvx2)
        val5 = reduce(np.dot, [yvx, xvx, yvkvkvx2.T])
        
        xvkvkvx2 = cal_xvkvkvx_res(xmat, vinv, onei, onej, leg_tp, kin_dct, var_ind, m)
        if var_ind[m, 1] != var_ind[m, 2]:
            xvkvkvx2 = xvkvkvx2 + cal_xvkvkvx_res(xmat, vinv, onej, onei, leg_tp, kin_dct, var_ind, m)
        val6 = reduce(np.dot, [yvx, xvx, xvkvkvx2, xvx, yvx.T])
        
        val7 = reduce(np.dot, [yvx, xvx, xvkvx, xvx, yvvx.T])
        val8 = reduce(np.dot, [yvx, xvx, xvkvx, xvx, xvvx, xvx, yvx.T])
        temp = val1 - val2 - val3 + val4 - val5 + val6 + val7 - val8
        ai_mat[-1, m] = 0.5 * np.sum(temp)
    
    # tr(vk)
    trvk = np.trace(reduce(np.add, vinv))
    # tr(pk)
    trpk = np.sum(np.multiply(xvx, xvvx))
    # ypkpy
    yvvy = reduce(np.matmul, [yv, yv.transpose(0, 2, 1)])
    yvvy = reduce(np.add, yvvy)
    ypkpy2 = reduce(np.dot, [yvx, xvx, yvvx.T])
    ypkpy3 = np.dot(yvx, xvx)
    ypkpy3 = reduce(np.dot, [ypkpy3, xvvx, ypkpy3.T])
    ypkpy = np.sum(yvvy) - 2 * np.sum(ypkpy2) + np.sum(ypkpy3)
    fd_mat[-1] = -0.5 * (trvk - trpk) + 0.5 * ypkpy
    
    # ai_mat
    yvvvy = reduce(np.matmul, [yv, vinv, yv.transpose(0, 2, 1)])
    val1 = reduce(np.add, yvvvy)
    yvvvx = reduce(np.matmul, [yv, vinv, vinv, xmat])
    yvvvx = reduce(np.add, yvvvx)
    val2 = reduce(np.dot, [yvvvx, xvx, yvx.T])
    val3 = reduce(np.dot, [yvvx, xvx, yvvx.T])
    val4 = reduce(np.dot, [yvvx, xvx, xvvx, xvx, yvx.T])
    val5 = reduce(np.dot, [yvx, xvx, yvvvx.T])
    xvvvx = reduce(np.matmul, [xmat.transpose(0, 2, 1), vinv, vinv, vinv, xmat])
    xvvvx = reduce(np.add, xvvvx)
    val6 = reduce(np.dot, [yvx, xvx, xvvvx, xvx, yvx.T])
    val7 = reduce(np.dot, [yvx, xvx, xvvx, xvx, yvvx.T])
    val8 = reduce(np.dot, [yvx, xvx, xvvx, xvx, xvvx, xvx, yvx.T])
    temp = val1 - val2 - val3 + val4 - val5 + val6 + val7 - val8
    ai_mat[-1, -1] = 0.5 * np.sum(temp)
    
    ai_mat = ai_mat + np.tril(ai_mat, k=-1).T
    
    return fd_mat, ai_mat


def pre_fdai_mat_eigen_glm(y, xmat, leg_tp, kin_eigen_val, cov_add, cov_per, cov_res, var_com, var_ind, kin_dct):
    vinv = np.multiply(kin_eigen_val.reshape(len(kin_eigen_val), 1, 1), tri_matT(leg_tp, cov_add)) + \
           tri_matT(leg_tp, cov_per) + np.diag([np.sum(cov_res)] * leg_tp.shape[0])
    vinv = np.linalg.inv(vinv)
    
    xvx = reduce(np.matmul, [xmat.transpose(0, 2, 1), vinv, xmat])
    xvx = reduce(np.add, xvx)
    xvx = np.linalg.inv(xvx)
    
    # xvy
    xvy = np.matmul(xmat.transpose(0, 2, 1), np.matmul(vinv, y))
    xvy = reduce(np.add, xvy)
    
    # py
    y_xb = y - np.matmul(xmat, np.dot(xvx, xvy))
    py = np.matmul(vinv, y_xb)
    
    fd_mat = np.array([0.0] * len(var_com))
    ai_mat = np.array([0.0] * len(var_com) * len(var_com)).reshape(len(var_com), len(var_com))
    ai_kpy = []
    ai_pkpy = []
    for m in range(var_ind.shape[0] - 1):
        
        onei = np.array([0.0] * leg_tp.shape[1]).reshape(leg_tp.shape[1], 1)
        onej = onei.copy()
        onei[var_ind[m, 1], 0] = 1.0
        onej[var_ind[m, 2], 0] = 1.0
        one = np.dot(onei, onej.T)
        one = one + one.T - np.diag(np.diag(one))
        tit = reduce(np.dot, [leg_tp, one, leg_tp.T])
        
        # tr(vk)
        trvk = reduce(np.matmul, [onej.T, leg_tp.T, vinv, leg_tp, onei])
        trvk = np.multiply(kin_dct[var_ind[m, 0]], trvk)
        trvk = np.sum(reduce(np.add, trvk))
        
        if var_ind[m, 1] != var_ind[m, 2]:
            trvk = 2 * trvk
        
        # tr(pk)
        temp2 = reduce(np.matmul, [onej.T, leg_tp.T, vinv, xmat])
        temp1 = reduce(np.matmul, [onei.T, leg_tp.T, vinv, xmat])
        xvkvx = np.matmul(temp1.transpose(0, 2, 1), temp2)
        xvkvx = np.multiply(kin_dct[var_ind[m, 0]], xvkvx)
        xvkvx = reduce(np.add, xvkvx)
        if var_ind[m, 1] != var_ind[m, 2]:
            xvkvx = xvkvx + xvkvx.T
        trpk = np.sum(np.multiply(xvx, xvkvx))
        
        kpy = np.matmul(tit, py)
        kpy = np.multiply(kin_dct[var_ind[m, 0]], kpy)
        ai_kpy.append(kpy)
        
        ypkpy = np.sum(np.matmul(py.transpose(0, 2, 1), kpy))
        fd_mat[m] = -0.5 * (trvk - trpk) + 0.5 * ypkpy
        
        xvkpy = np.matmul(xmat.transpose(0, 2, 1), np.matmul(vinv, kpy))
        xvkpy = reduce(np.add, xvkpy)
        
        kpy_xb = kpy - np.matmul(xmat, np.dot(xvx, xvkpy))
        pkpy = np.matmul(vinv, kpy_xb)
        ai_pkpy.append(pkpy)
    
    trvk = np.sum(map(np.trace, vinv))
    # tr(pk)
    xvvx = reduce(np.matmul, [xmat.transpose(0, 2, 1), vinv, vinv, xmat])
    xvvx = reduce(np.add, xvvx)
    trpk = np.sum(np.multiply(xvx, xvvx))
    ai_kpy.append(py)
    
    ypkpy = np.sum(np.matmul(py.transpose(0, 2, 1), py))
    fd_mat[-1] = -0.5 * (trvk - trpk) + 0.5 * ypkpy
    
    xvkpy = np.matmul(xmat.transpose(0, 2, 1), np.matmul(vinv, py))
    xvkpy = reduce(np.add, xvkpy)
    
    kpy_xb = py - np.matmul(xmat, np.dot(xvx, xvkpy))
    pkpy = np.matmul(vinv, kpy_xb)
    ai_pkpy.append(pkpy)
    
    for i in range(len(ai_kpy)):
        for j in range(i + 1):
            ai_mat[i, j] = np.sum(np.matmul(ai_kpy[i].transpose(0, 2, 1), ai_pkpy[j]))
    
    ai_mat = ai_mat + np.tril(ai_mat, k=-1).T
    ai_mat = 0.5 * ai_mat
    
    return fd_mat, ai_mat

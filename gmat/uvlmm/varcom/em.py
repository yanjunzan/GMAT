import numpy as np
from functools import reduce
import gc
import logging
from scipy.sparse import hstack


def em_mme(y, xmat, zmat_lst, gmat_inv_lst, init=None, maxiter=100, cc_par=1.0e-8):
    """
    Estimate variance parameters with mme based em
    :param y:
    :param xmat:
    :param gmat_inv_lst:
    :param init:
    :param maxiter:
    :param cc_par:
    :param cc_gra:
    :return:
    """
    num_varcom = len(gmat_inv_lst) + 1
    var_com = np.ones(num_varcom)
    if init is not None:
        var_com = np.array(init)
    var_com_new = np.array(var_com)
    logging.info("Prepare the coef matrix without variance parameters")
    xmat_df = xmat.shape[1]
    zmat_concat = hstack(zmat_lst)
    xzmat = hstack([xmat, zmat_concat])
    coef_null = xzmat.T.dot(xzmat)
    rhs_null = xzmat.T.dot(y)
    logging.info("Start iteration")
    iter = 0
    cc_par_val = 10000.0
    while iter < maxiter:
        iter += 1
        logging.info('Round: {:d}'.format(iter))
        # coef matrix
        coef = coef_null.toarray() / var_com[-1]
        df_sum = xmat_df
        for k in range(len(gmat_inv_lst)):
            indexa = df_sum
            df_sum += gmat_inv_lst[k].shape[0]
            indexb = df_sum
            coef[indexa:indexb, indexa:indexb] = coef[indexa:indexb, indexa:indexb] + gmat_inv_lst[k]/var_com[k]
        # right hand
        rhs_mat = rhs_null/var_com[-1]
        coef_inv = np.linalg.inv(coef)
        eff = np.dot(coef_inv, rhs_mat)
        # error variance
        e_hat = y - xzmat.dot(eff)
        var_com_new[-1] = np.sum(np.dot(e_hat.T, e_hat)) + np.sum((xzmat.T.dot(xzmat)).toarray() * coef_inv) # fast trace calculation
        var_com_new[-1] /= y.shape[0]
        # random variances
        df_sum = xmat_df
        for k in range(len(gmat_inv_lst)):
            indexa = df_sum
            df_sum += gmat_inv_lst[k].shape[0]
            indexb = df_sum
            var_com_new[k] = np.sum(coef_inv[indexa:indexb, indexa:indexb]*gmat_inv_lst[k]) \
                             + np.sum(reduce(np.dot, [eff[indexa:indexb, :].T, gmat_inv_lst[k], eff[indexa:indexb, :]]))
            qk = gmat_inv_lst[k].shape[0]
            var_com_new[k] /= qk
        # cc
        delta = np.array(var_com_new) - np.array(var_com)
        cc_par_val = np.sum(delta*delta)/np.sum(np.array(var_com_new)*np.array(var_com_new))
        cc_par_val = np.sqrt(cc_par_val)
        var_com = np.array(var_com_new)
        logging.info('Norm of update vector: {:e}'.format(cc_par_val))
        var_com_str = ' '.join(np.array(var_com, dtype=str))
        logging.info('Updated variances: {}'.format(var_com_str))
        if cc_par_val < cc_par:
            break
    if cc_par_val < cc_par:
        logging.info('Variances converged.')
    else:
        logging.info('Variances not converged.')
    return var_com


def em_vmat(y, xmat, zmat_lst, gmat_lst, init=None, maxiter=100, cc_par=1.0e-8):
    """
    :param y:
    :param xmat:
    :param zmat_lst:
    :param gmat_lst:
    :param init:
    :param maxiter:
    :param cc_par:
    :return:
    """
    logging.info("#####Prepare#####")
    var_com = [1.0] * (len(gmat_lst) + 1)
    if init is not None:
        var_com = init[:]
    var_com = np.array(var_com)
    var_com_new = np.array(var_com)
    y = np.array(y).reshape(-1, 1)
    n = y.shape[0]
    xmat = np.array(xmat).reshape(n, -1)
    zgmat_lst = []
    for val in range(len(gmat_lst)):
        zgmat_lst.append(zmat_lst[val].dot((zmat_lst[val].dot(gmat_lst[val])).T))
    logging.info('Initial variances: ' + ' '.join(list(np.array(var_com_new, dtype=str))))
    logging.info("#####Start the iteration#####\n\n")
    iter = 0
    delta = 1000.0
    cc_gra_val = 1000.0
    cc_par_val = 1000.0
    while iter < maxiter:
        iter += 1
        logging.info('###Round: ' + str(iter) + '###')
        vmat = np.diag([var_com[-1]] * n)
        for val in range(len(zgmat_lst)):
            vmat += zgmat_lst[val] * var_com[val]
        ll_v = np.linalg.slogdet(vmat)[1]
        vmat = np.linalg.inv(vmat)
        # 计算P矩阵
        vxmat = np.dot(vmat, xmat)
        xvxmat = np.dot(xmat.T, vxmat)
        ll_xvx = np.linalg.slogdet(xvxmat)[1]
        xvxmat = np.linalg.inv(xvxmat)
        pmat = reduce(np.dot, [vxmat, xvxmat, vxmat.T])
        pmat = vmat - pmat
        pymat = np.dot(pmat, y)
        # -2logL
        ll_ypy = np.sum(np.dot(y.T, pymat))
        ll_val = -2 * (ll_v + ll_xvx + ll_ypy)
        del vmat, vxmat, xvxmat
        gc.collect()
        # 计算一阶偏导、工具变量
        fd_mat = []
        wv_mat = []
        for val in range(len(zgmat_lst)):
            fd_mat_val = -np.trace(np.dot(pmat, zgmat_lst[val])) + reduce(np.dot, [pymat.T, zgmat_lst[val], pymat])
            fd_mat.append(0.5 * np.sum(fd_mat_val))
            wv_mat.append(np.dot(zgmat_lst[val], pymat))
        fd_mat_val = -np.trace(pmat) + np.dot(pymat.T, pymat)
        fd_mat.append(0.5 * np.sum(fd_mat_val))
        fd_mat = np.array(fd_mat)
        wv_mat.append(pymat)
        wv_mat = np.concatenate(wv_mat, axis=1)
        # AI矩阵
        ai_mat = 0.5 * reduce(np.dot, [wv_mat.T, pmat, wv_mat])
        del pymat, wv_mat, pmat
        gc.collect()
        em_mat_inv = []
        for val in var_com:
            em_mat_inv.append((2 * val * val)/n)
        em_mat_inv = np.diag(em_mat_inv)
        delta = np.dot(em_mat_inv, fd_mat)
        var_com_new = var_com + delta
        cc_par_val = np.sum(delta * delta) / np.sum(var_com_new * var_com_new)
        cc_par_val = np.sqrt(cc_par_val)
        var_com = var_com_new[:]
        cc_gra_val = np.sqrt(np.sum(fd_mat * fd_mat))
        logging.info('-2logL for last iteration: ' + str(ll_val))
        logging.info('Norm of gradient vector for last iteration: ' + str(cc_gra_val))
        logging.info('Norm of update vector: ' + str(cc_par_val))
        logging.info('Updated variances: ' + ' '.join(list(np.array(var_com_new, dtype=str))))
        if cc_gra_val < cc_gra and cc_par_val < cc_par:
            break
    if cc_gra_val < cc_gra and cc_par_val < cc_par:
        logging.info('Variances converged.')
    else:
        logging.info('Variances not converged.')
    return var_com

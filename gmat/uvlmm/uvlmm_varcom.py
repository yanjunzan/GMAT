import numpy as np
from functools import reduce
import gc
import logging


def wemai_multi_gmat(y, xmat, zmat, gmat_lst, init=None, maxiter=200, cc_par=1.0e-8, cc_gra=1.0e-6):
    """
    Estimate variances for univariate linear mixed model. Multiple genomic relationship matrixes can be included.
    Weighted EM an AI algorithm are used.
    :param y: phenotypic vectors
    :param xmat: designed matrix for fixed effect.
    :param zmat: designed matrix for random effets，csr sparse matrix. y, xmat and zmat can be prepared with
    design_matrix_wemai_multi_gmat program.
    :param gmat_lst: a list of genomic relationship matrixes.
    :param init: initial values for variances. default value is None.
    :param maxiter: the maximal number of interactions. default value is 200.
    :param cc_par: The convergence criteria for update vector.
    :param cc_gra: The convergence criteria for gradient vector
    :return: the estimated variances
    """
    logging.info("#####Prepare#####")
    var_com = [1.0]*(len(gmat_lst)+1)
    if init is not None:
        var_com = init[:]
    var_com = np.array(var_com)
    var_com_new = var_com[:]
    y = np.array(y).reshape(-1, 1)
    n = y.shape[0]
    xmat = np.array(xmat).reshape(n, -1)
    zgmat_lst = []
    for val in range(len(gmat_lst)):
        zgmat_lst.append(zmat.dot((zmat.dot(gmat_lst[val])).T))
    logging.info('Initial variances: ' + ' '.join(list(np.array(var_com_new, dtype=str))))
    logging.info("#####Start the iteration#####\n\n")
    iter = 0
    delta = 1000.0
    cc_gra_val = 1000.0
    cc_par_val = 1000.0
    while iter < maxiter:
        iter += 1
        logging.info('###Round: ' + str(iter) + '###')
        vmat = np.diag([var_com[-1]]*n)
        for val in range(len(zgmat_lst)):
            vmat += zgmat_lst[val]*var_com[val]
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
        ll_val = -2*(ll_v + ll_xvx + ll_ypy)
        del vmat, vxmat, xvxmat
        gc.collect()
        # 计算一阶偏导、工具变量
        fd_mat = []
        wv_mat = []
        for val in range(len(zgmat_lst)):
            fd_mat_val = -np.trace(np.dot(pmat, zgmat_lst[val])) + reduce(np.dot, [pymat.T, zgmat_lst[val], pymat])
            fd_mat.append(0.5*np.sum(fd_mat_val))
            wv_mat.append(np.dot(zgmat_lst[val], pymat))
        fd_mat_val = -np.trace(pmat) + np.dot(pymat.T, pymat)
        fd_mat.append(0.5 * np.sum(fd_mat_val))
        fd_mat = np.array(fd_mat)
        wv_mat.append(pymat)
        wv_mat = np.concatenate(wv_mat, axis=1)
        # AI矩阵
        ai_mat = 0.5*reduce(np.dot, [wv_mat.T, pmat, wv_mat])
        del pymat, wv_mat, pmat
        gc.collect()
        em_mat = []
        for val in var_com:
            em_mat.append(n/(val*val))
        em_mat = np.diag(em_mat)
        for j in range(0, 101):
            weight = j * 0.01
            wemai_mat = (1 - weight) * ai_mat + weight * em_mat
            delta = np.dot(np.linalg.inv(wemai_mat), fd_mat)
            var_com_new = var_com + delta
            if min(var_com_new) > 0:
                logging.info('EM weight value: ' + str(weight))
                break
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


def em_mme(y, xmat, gmat_inv, init=None, maxiter=100, cc=1.0e-8):
    var_com = [1.0, 1.0]
    if init is not None:
        var_com = init[:]
    var_com_new = var_com[:]
    # 准备不含方差组分的系数矩阵
    xmat_df, gmat_inv_df = xmat.shape[1], gmat_inv.shape[0]
    y_df = len(y)
    coef_pre = np.identity(xmat_df+gmat_inv_df)
    coef_pre[:xmat_df, :xmat_df] = np.dot(xmat.T, xmat)
    coef_pre[:xmat_df, xmat_df:] = xmat.T
    coef_pre[xmat_df:, :xmat_df] = xmat
    w_mat = np.concatenate((xmat, np.identity(gmat_inv_df)), axis=1)
    # 开始迭代
    iter = 0
    cc_val = 100.0
    while iter < maxiter:
        iter += 1
        print('Round: ', iter)
        # 系数矩阵
        coef = coef_pre/var_com[1]
        coef[xmat_df:, xmat_df:] = coef[xmat_df:, xmat_df:] + gmat_inv/var_com[0]
        # 右手项
        rhs_mat = np.dot(w_mat.T, y)/var_com[1]
        coef_inv = np.linalg.inv(coef)
        eff = np.dot(coef_inv, rhs_mat)
        var_com_new[0] = np.dot(eff[xmat_df:], np.dot(gmat_inv, eff[xmat_df:])) + \
                     np.trace(np.dot(gmat_inv, coef_inv[xmat_df:, xmat_df:]))
        var_com_new[0] = var_com_new[0]/gmat_inv_df
        e_hat = y - np.dot(xmat, eff[:xmat_df]) - eff[xmat_df:]
        var_com_new[1] = np.dot(e_hat, e_hat) + np.trace(np.dot(w_mat, np.dot(coef_inv, w_mat.T)))
        var_com_new[1] = var_com_new[1]/y_df
        print('Updated variances:', var_com_new)
        delta = np.array(var_com_new) - np.array(var_com)
        cc_val = np.sum(delta*delta)/np.sum(np.array(var_com_new)*np.array(var_com_new))
        cc_val = np.sqrt(cc_val)
        var_com = var_com_new[:]
        if cc_val < cc:
            break
    if cc_val < cc:
        print('Variances converged.')
    else:
        print('Variances not converged.')
    return var_com


def pxem_mme(y, xmat, gmat_inv, init=None, maxiter=100, cc=1.0e-8):
    var_com = [1.0, 1.0]
    if init is not None:
        var_com = init[:]
    var_com_new = var_com[:]
    # 准备不含方差组分的系数矩阵
    xmat_df, gmat_inv_df = xmat.shape[1], gmat_inv.shape[0]
    y_df = len(y)
    coef_pre = np.identity(xmat_df+gmat_inv_df)
    coef_pre[:xmat_df, :xmat_df] = np.dot(xmat.T, xmat)
    coef_pre[:xmat_df, xmat_df:] = xmat.T
    coef_pre[xmat_df:, :xmat_df] = xmat
    w_mat = np.concatenate((xmat, np.identity(gmat_inv_df)), axis=1)
    # 开始迭代
    iter = 0
    cc_val = 100.0
    while iter < maxiter:
        iter += 1
        print('Round: ', iter)
        # 系数矩阵
        coef = coef_pre/var_com[1]
        coef[xmat_df:, xmat_df:] = coef[xmat_df:, xmat_df:] + gmat_inv/var_com[0]
        # 右手项
        rhs_mat = np.dot(w_mat.T, y)/var_com[1]
        coef_inv = np.linalg.inv(coef)
        eff = np.dot(coef_inv, rhs_mat)
        var_com_new[0] = np.dot(eff[xmat_df:], np.dot(gmat_inv, eff[xmat_df:])) + \
                     np.trace(np.dot(gmat_inv, coef_inv[xmat_df:, xmat_df:]))
        var_com_new[0] = var_com_new[0]/gmat_inv_df
        e_hat = y - np.dot(xmat, eff[:xmat_df]) - eff[xmat_df:]
        var_com_new[1] = np.dot(e_hat, e_hat) + np.trace(np.dot(w_mat, np.dot(coef_inv, w_mat.T)))
        var_com_new[1] = var_com_new[1]/y_df
        gamma1 = np.dot(eff[xmat_df:], y - np.dot(xmat, eff[:xmat_df])) - np.trace(np.dot(xmat, coef_inv[:xmat_df, xmat_df:]))
        gamma2 = np.dot(eff[xmat_df:], eff[xmat_df:]) + np.trace(coef_inv[xmat_df:, xmat_df:])
        gamma = gamma1/gamma2
        var_com_new[0] = var_com_new[0]*gamma*gamma
        print('Updated variances:', var_com_new)
        delta = np.array(var_com_new) - np.array(var_com)
        cc_val = np.sum(delta*delta)/np.sum(np.array(var_com_new)*np.array(var_com_new))
        cc_val = np.sqrt(cc_val)
        var_com = var_com_new[:]
        if cc_val < cc:
            break
    if cc_val < cc:
        print('Variances converged.')
    else:
        print('Variances not converged.')
    return var_com


def ai_mme(y, xmat, gmat_inv, init=None, maxiter=100, cc=1.0e-8):
    var_com = [1.0, 1.0]
    if init is not None:
        var_com = init[:]
    var_com = np.array(var_com)
    # 准备不含方差组分的系数矩阵
    xmat_df, gmat_inv_df = xmat.shape[1], gmat_inv.shape[0]
    y_df = len(y)
    coef_pre = np.identity(xmat_df + gmat_inv_df)
    coef_pre[:xmat_df, :xmat_df] = np.dot(xmat.T, xmat)
    coef_pre[:xmat_df, xmat_df:] = xmat.T
    coef_pre[xmat_df:, :xmat_df] = xmat
    w_mat = np.concatenate((xmat, np.identity(gmat_inv_df)), axis=1)
    # 开始迭代
    iter = 0
    cc_val = 100.0
    fd_mat = np.zeros(2)
    # ai_mat = np.zeros((2, 2))
    while iter < maxiter:
        iter += 1
        print('Round: ', iter)
        # 系数矩阵
        coef = coef_pre/var_com[1]
        coef[xmat_df:, xmat_df:] = coef[xmat_df:, xmat_df:] + gmat_inv/var_com[0]
        # 右手项
        rhs_mat = np.dot(w_mat.T, y)/var_com[1]
        coef_inv = np.linalg.inv(coef)
        eff = np.dot(coef_inv, rhs_mat)
        e_hat = y - np.dot(xmat, eff[:xmat_df]) - eff[xmat_df:]
        fd_mat[0] = gmat_inv_df/var_com[0] - np.trace(np.dot(coef_inv[xmat_df:, xmat_df:], gmat_inv))/(var_com[0]*var_com[0]) - \
            np.dot(eff[xmat_df:], np.dot(gmat_inv, eff[xmat_df:]))/(var_com[0]*var_com[0])
        fd_mat[1] = y_df/var_com[1] - np.trace(np.dot(np.dot(coef_inv, w_mat.T), w_mat))/(var_com[1]*var_com[1]) - \
            np.dot(e_hat, e_hat)/(var_com[1]*var_com[1])
        fd_mat = -0.5*fd_mat
        h_mat1 = np.array(eff[xmat_df:]/var_com[0]).reshape(gmat_inv_df, 1)
        h_mat2 = np.array(e_hat / var_com[1]).reshape(y_df, 1)
        h_mat = np.concatenate((h_mat1, h_mat2), axis=1)
        qrq = np.divide(np.dot(h_mat.T, h_mat), var_com[-1])
        left = np.divide(np.dot(w_mat.T, h_mat), var_com[-1])
        eff_h_mat = np.dot(coef_inv, left)
        ai_mat = np.subtract(qrq, np.dot(left.T, eff_h_mat))
        ai_mat = 0.5 * ai_mat
        ai_mat_inv = np.linalg.inv(ai_mat)
        var_com_new = var_com + np.dot(ai_mat_inv, fd_mat)
        print('Updated variances:', var_com_new)
        delta = np.array(var_com_new) - np.array(var_com)
        cc_val = np.sum(delta * delta) / np.sum(np.array(var_com_new) * np.array(var_com_new))
        cc_val = np.sqrt(cc_val)
        var_com = var_com_new[:]
        if cc_val < cc:
            break
    if cc_val < cc:
        print('Variances converged.')
    else:
        print('Variances not converged.')
    return var_com


def emai_mme(y, xmat, gmat_inv, init=None, maxiter=100, cc=1.0e-8):
    var_com = [1.0, 1.0]
    if init is not None:
        var_com = init[:]
    var_com = np.array(var_com)
    var_com_new = var_com[:]
    # 准备不含方差组分的系数矩阵
    xmat_df, gmat_inv_df = xmat.shape[1], gmat_inv.shape[0]
    y_df = len(y)
    coef_pre = np.identity(xmat_df + gmat_inv_df)
    coef_pre[:xmat_df, :xmat_df] = np.dot(xmat.T, xmat)
    coef_pre[:xmat_df, xmat_df:] = xmat.T
    coef_pre[xmat_df:, :xmat_df] = xmat
    w_mat = np.concatenate((xmat, np.identity(gmat_inv_df)), axis=1)
    # 开始迭代
    iter = 0
    cc_val = 100.0
    fd_mat = np.zeros(2)
    em_mat = np.zeros((2, 2))
    while iter < maxiter:
        iter += 1
        print('Round: ', iter)
        # 系数矩阵
        coef = coef_pre/var_com[1]
        coef[xmat_df:, xmat_df:] = coef[xmat_df:, xmat_df:] + gmat_inv/var_com[0]
        # 右手项
        rhs_mat = np.dot(w_mat.T, y)/var_com[1]
        coef_inv = np.linalg.inv(coef)
        eff = np.dot(coef_inv, rhs_mat)
        e_hat = y - np.dot(xmat, eff[:xmat_df]) - eff[xmat_df:]
        fd_mat[0] = gmat_inv_df/var_com[0] - np.trace(np.dot(coef_inv[xmat_df:, xmat_df:], gmat_inv))/(var_com[0]*var_com[0]) - \
            np.dot(eff[xmat_df:], np.dot(gmat_inv, eff[xmat_df:]))/(var_com[0]*var_com[0])
        fd_mat[1] = y_df/var_com[1] - np.trace(np.dot(np.dot(coef_inv, w_mat.T), w_mat))/(var_com[1]*var_com[1]) - \
            np.dot(e_hat, e_hat)/(var_com[1]*var_com[1])
        fd_mat = -0.5*fd_mat
        h_mat1 = np.array(eff[xmat_df:]/var_com[0]).reshape(gmat_inv_df, 1)
        h_mat2 = np.array(e_hat / var_com[1]).reshape(y_df, 1)
        h_mat = np.concatenate((h_mat1, h_mat2), axis=1)
        qrq = np.divide(np.dot(h_mat.T, h_mat), var_com[-1])
        left = np.divide(np.dot(w_mat.T, h_mat), var_com[-1])
        eff_h_mat = np.dot(coef_inv, left)
        ai_mat = np.subtract(qrq, np.dot(left.T, eff_h_mat))
        ai_mat = 0.5 * ai_mat
        print(fd_mat, ai_mat)
        em_mat[0, 0] = gmat_inv_df / (var_com[0] * var_com[0])
        em_mat[1, 1] = y_df / (var_com[1] * var_com[1])
        for j in range(0, 51):
            weight = j * 0.1
            wemai_mat = (1 - weight) * ai_mat + weight * em_mat
            delta = np.dot(np.linalg.inv(wemai_mat), fd_mat)
            var_com_new = var_com + delta
            if min(var_com_new) > 0:
                print('EM weight value:', weight)
                break
        print('Updated variances:', var_com_new)
        delta = np.array(var_com_new) - np.array(var_com)
        cc_val = np.sum(delta * delta) / np.sum(np.array(var_com_new) * np.array(var_com_new))
        cc_val = np.sqrt(cc_val)
        var_com = var_com_new[:]
        if cc_val < cc:
            break
    if cc_val < cc:
        print('Variances converged.')
    else:
        print('Variances not converged.')
    return var_com


def pxemai_mme(y, xmat, gmat_inv, init=None, maxiter=100, cc=1.0e-8):
    var_com = [1.0, 1.0]
    if init is not None:
        var_com = init[:]
    var_com = np.array(var_com)
    var_com_new = var_com[:]
    # 准备不含方差组分的系数矩阵
    xmat_df, gmat_inv_df = xmat.shape[1], gmat_inv.shape[0]
    y_df = len(y)
    coef_pre = np.identity(xmat_df + gmat_inv_df)
    coef_pre[:xmat_df, :xmat_df] = np.dot(xmat.T, xmat)
    coef_pre[:xmat_df, xmat_df:] = xmat.T
    coef_pre[xmat_df:, :xmat_df] = xmat
    w_mat = np.concatenate((xmat, np.identity(gmat_inv_df)), axis=1)
    # 开始迭代
    iter = 0
    cc_val = 100.0
    weight = 0.0
    fd_mat = np.zeros(2)
    em_mat = np.zeros((2, 2))
    while iter < maxiter:
        iter += 1
        print('Round: ', iter)
        # 系数矩阵
        coef = coef_pre/var_com[1]
        coef[xmat_df:, xmat_df:] = coef[xmat_df:, xmat_df:] + gmat_inv/var_com[0]
        # 右手项
        rhs_mat = np.dot(w_mat.T, y)/var_com[1]
        coef_inv = np.linalg.inv(coef)
        eff = np.dot(coef_inv, rhs_mat)
        e_hat = y - np.dot(xmat, eff[:xmat_df]) - eff[xmat_df:]
        fd_mat[0] = gmat_inv_df/var_com[0] - np.trace(np.dot(coef_inv[xmat_df:, xmat_df:], gmat_inv))/(var_com[0]*var_com[0]) - \
            np.dot(eff[xmat_df:], np.dot(gmat_inv, eff[xmat_df:]))/(var_com[0]*var_com[0])
        fd_mat[1] = y_df/var_com[1] - np.trace(np.dot(np.dot(coef_inv, w_mat.T), w_mat))/(var_com[1]*var_com[1]) - \
            np.dot(e_hat, e_hat)/(var_com[1]*var_com[1])
        fd_mat = -0.5*fd_mat
        h_mat1 = np.array(eff[xmat_df:]/var_com[0]).reshape(gmat_inv_df, 1)
        h_mat2 = np.array(e_hat / var_com[1]).reshape(y_df, 1)
        h_mat = np.concatenate((h_mat1, h_mat2), axis=1)
        qrq = np.divide(np.dot(h_mat.T, h_mat), var_com[-1])
        left = np.divide(np.dot(w_mat.T, h_mat), var_com[-1])
        eff_h_mat = np.dot(coef_inv, left)
        ai_mat = np.subtract(qrq, np.dot(left.T, eff_h_mat))
        ai_mat = 0.5 * ai_mat
        print(fd_mat, ai_mat)
        em_mat[0, 0] = gmat_inv_df / (var_com[0] * var_com[0])
        em_mat[1, 1] = y_df / (var_com[1] * var_com[1])
        for j in range(0, 51):
            weight = j * 0.1
            wemai_mat = (1 - weight) * ai_mat + weight * em_mat
            delta = np.dot(np.linalg.inv(wemai_mat), fd_mat)
            var_com_new = var_com + delta
            if min(var_com_new) > 0:
                print('EM weight value:', weight)
                break
        if weight > 0.001:
            gamma1 = np.dot(eff[xmat_df:], y - np.dot(xmat, eff[:xmat_df])) - np.trace(
                np.dot(xmat, coef_inv[:xmat_df, xmat_df:]))
            gamma2 = np.dot(eff[xmat_df:], eff[xmat_df:]) + np.trace(coef_inv[xmat_df:, xmat_df:])
            gamma = gamma1 / gamma2
            var_com_new[0] = var_com_new[0] * gamma * gamma
        print('Updated variances:', var_com_new)
        delta = np.array(var_com_new) - np.array(var_com)
        cc_val = np.sum(delta * delta) / np.sum(np.array(var_com_new) * np.array(var_com_new))
        cc_val = np.sqrt(cc_val)
        var_com = var_com_new[:]
        if cc_val < cc:
            break
    if cc_val < cc:
        print('Variances converged.')
    else:
        print('Variances not converged.')
    return var_com


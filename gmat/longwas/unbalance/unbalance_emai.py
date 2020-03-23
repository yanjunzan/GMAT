import math
import numpy as np
import pandas as pd
from scipy import linalg
from scipy import sparse
from scipy.sparse import csr_matrix, isspmatrix, hstack, vstack


from .unbalance_pre_mat import pre_covar_inv, pre_coef_mat
from .unbalance_iter_mat import unbalance_pre_fd_mat, unbalance_pre_em_mat, unbalance_pre_ai_mat


def unbalance_emai(y, xmat, zmat, rmat_inv_lst, fix_const=False, ran_const=False, init=None, max_iter=30, cc_par=1.0e-8,
                   cc_gra=1.0e-6, cc_ll=0.001, em_weight_step=0.02):
    num_var_diag = 1  # the number of variances (not include non-diag)
    for i in range(len(zmat)):
        num_var_diag += len(zmat[i])
    y_var = np.var(y)/num_var_diag  # for the initial values of variances
    
    num_record = y.shape[0]
    var_com = []
    eff_ind = [[0, xmat.shape[1]]]  # the index for all effects [start end]
    zmat_concat_lst = []  # combined random matrix
    covar_dim = []  # the dim for covariance matrix
    vari = []
    varij = []
    varik = []
    for i in range(len(zmat)):
        temp = [eff_ind[i][-1]]
        zmat_concat_lst.append(hstack(zmat[i]))
        covar_dim.append(len(zmat[i]))
        for j in range(len(zmat[i])):
            temp.append(temp[-1] + zmat[i][j].shape[1])
            for k in range(j + 1):
                vari.append(i+1)
                varij.append(j+1)
                varik.append(k+1)
                if j == k:
                    var_com.append(y_var)
                else:
                    var_com.append(0.0)
        eff_ind.append(temp)
    var_com.append(y_var)
    vari.append(len(zmat) + 1)
    varij.append(1)
    varik.append(1)
    
    if init is None:
        var_com = np.array(var_com)
    else:
        if len(var_com) != len(init):
            print('The length of initial variances should be', len(var_com))
            exit()
        else:
            var_com = np.array(init)
    var_com_update = var_com * 10
    delta = var_com_update - var_com
    
    print('***prepare the MME**')
    zmat_concat = hstack(zmat_concat_lst)  # design matrix for random effects
    wmat = hstack([xmat, zmat_concat])  # merged design matrix
    coef_mat_pure = np.dot(wmat.T, wmat)  # C matrix
    rhs_pure = wmat.T.dot(y)  # right hand
    
    # em weight vector
    if em_weight_step <= 0.0 or em_weight_step > 1.0:
        print('The em weight step should be between 0 (not include) and 1 (include)')
        exit()
    
    iter_count = 0
    
    cc_par_val = 1000.0
    cc_gra_val = 1000.0
    ll_val = 1000.0
    cc_ll_val = 1000.0
    ll_val_pre = 1.0e30
    print("initial variances:", ' '.join(np.array(var_com, dtype=str)))
    
    covar_inv = pre_covar_inv(covar_dim, var_com)
    if covar_inv is None:
        print("Initial variances is not positive define, please check!")
        exit()
    coef_mat_inv, eff, e = pre_coef_mat(y, xmat, eff_ind, zmat_concat, covar_dim, var_com, covar_inv, coef_mat_pure,
                                        rmat_inv_lst, rhs_pure)
    
    while iter_count < max_iter:
        iter_count += 1
        print('***Start the iteration:', str(iter_count) + ' ***')
        # first-order derivative
        fd_mat = unbalance_pre_fd_mat(coef_mat_inv, rmat_inv_lst, covar_inv, eff, eff_ind, e, covar_dim,
                                      zmat_concat_lst, wmat, num_record, var_com)
        # AI matrix
        ai_mat = unbalance_pre_ai_mat(coef_mat_inv, covar_inv, eff, eff_ind, e, covar_dim, zmat_concat_lst, wmat, var_com)
        # EM matrix
        em_mat = unbalance_pre_em_mat(covar_dim, zmat_concat_lst, num_record, var_com)
        
        gamma = -em_weight_step
        while gamma < 1.0:
            gamma = gamma + em_weight_step
            if gamma >= 1.0:
                gamma = 1.0
            wemai_mat = (1 - gamma) * ai_mat + gamma * em_mat
            delta = np.dot(linalg.inv(wemai_mat), fd_mat)
            var_com_update = var_com + delta
            covar_inv = pre_covar_inv(covar_dim, var_com_update)
            if covar_inv is not None:
                print('EM weight value:', str(gamma))
                break
        
        print('Updated variances:', ' '.join(np.array(var_com_update, dtype=str)))
        if covar_inv is None:
            print("Updated variances is not positive define!")
            exit()
        
        coef_mat_inv, eff, e = pre_coef_mat(y, xmat, eff_ind, zmat_concat, covar_dim, var_com_update, covar_inv,
                                            coef_mat_pure, rmat_inv_lst, rhs_pure)
        
        # Convergence criteria
        cc_par_val = np.sum(pow(delta, 2)) / np.sum(pow(var_com_update, 2))
        cc_par_val = np.sqrt(cc_par_val)
        cc_gra_val = np.sqrt(np.sum(pow(fd_mat, 2))) / len(var_com)
        var_com = var_com_update.copy()
        print("Change in parameters, Norm of gradient vector:", str(cc_par_val) + ', ' + str(cc_gra_val))
       
        # Log likelihood function value
        yry = np.divide(np.dot(y.T, y), var_com[-1])
        left = np.divide(wmat.T.dot(y), var_com[-1])
        eff_y = np.dot(coef_mat_inv, left)
        ypy = np.sum(yry - np.dot(left.T, eff_y))
        det_c = -np.linalg.slogdet(coef_mat_inv)[1]
        det_r = num_record * np.log(var_com[-1])
        det_g = 0.0
        for i in range(len(covar_dim)):
            if isspmatrix(rmat_inv_lst[i]):
                det_g += -rmat_inv_lst[i].shape[1] * np.linalg.slogdet(covar_inv[i])[1]
            else:
                det_g += -rmat_inv_lst[i].shape[1] * np.linalg.slogdet(covar_inv[i])[1]
        ll_val = det_r + det_g + det_c + ypy
        print("-2logL: " + str(ll_val))
        cc_ll_val = abs(ll_val_pre - ll_val)
        ll_val_pre = ll_val
        if cc_par_val < cc_par and cc_gra_val < cc_gra and cc_ll_val < cc_ll:
            break
    
    if cc_par_val < cc_par and cc_gra_val < cc_gra and cc_ll_val < cc_ll:
        convergence = True
        print("Variances Converged")
    else:
        convergence = False
        print("Variances not Converged")
    
    # add
    fix_const_val = 0.0
    ran_const_val = 0.0
    if fix_const:
        temp = xmat.T.dot(xmat)
        fix_const_val += -np.linalg.slogdet(temp.toarray())[1]
    if ran_const:
        for i in range(len(covar_dim)):
            if not isspmatrix(rmat_inv_lst[i]):
                ran_const_val += -covar_dim[i] * np.linalg.slogdet(rmat_inv_lst[i])[1]
    ll_val = ll_val + fix_const_val + ran_const_val
    aic_val = 2.0*len(var_com) + ll_val
    print("AIC value: " + str(aic_val))
    bic_val = len(var_com)*math.log(num_record) + ll_val
    print("BIC value: " + str(bic_val))
    var_pd = {'vari': vari,
              "varij": varij,
              "varik": varik,
              "var_val": var_com}
    var_pd = pd.DataFrame(var_pd, columns=['vari', "varij", "varik", "var_val"])
    res = {'variances': var_pd,
           'cc_par': cc_par_val,
           'cc_gra': cc_par_val,
           'convergence': convergence,
           'll': ll_val,
           'effect': eff,
           'effect_ind': eff_ind,
           'coef_inv': coef_mat_inv,
           'AIC': aic_val,
           'BIC': bic_val}
    return res

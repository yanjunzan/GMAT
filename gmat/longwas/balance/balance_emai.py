import math
import numpy as np
from scipy import linalg
import datetime
import pandas as pd
import logging

from .common import *
from .pre_mat import *
from .iter_mat import *


def balance_emai(y, xmat, leg_tp, kin_eigen_val, init=None, maxiter=30, cc_par=1.0e-8, cc_gra=1.0e-6, em_weight_step=0.02):
    num_record = y.shape[0]*y.shape[1]*y.shape[2]
    ran_df = len(kin_eigen_val)
    cov_dim = leg_tp.shape[1]
    var_com = np.array(init)
    var_com_update = var_com.copy()
    delta = 1000.0
    logging.info("initial variances: " + ' '.join(np.array(var_com, dtype=str)))
    cov = pre_cov_mat_eigen(cov_dim, var_com)
    if cov is None:
        logging.info('The covariances are not positive defined!')
        exit()
    cov_add, cov_per, cov_res = cov[:]
    var_ind = np.tril_indices_from(cov_add)
    a = [0] * len(var_ind[0])
    a.extend([1]*len(var_ind[0]))
    a.append(2)
    b = list(var_ind[0])
    b.extend(b)
    b.append(0)
    c = list(var_ind[1])
    c.extend(c)
    c.append(0)
    var_ind = np.array([a, b, c]).T
    kin_dct = {
        0: kin_eigen_val.reshape(len(kin_eigen_val), 1, 1),
        1: 1.0,
        2: 1.0
    }
    iter_count = 0
    cc_par_val = 1000.0
    cc_gra_val = 1000.0
    while iter_count < maxiter:
        iter_count += 1
        logging.info('***Start the iteration: {} ***'.format(iter_count))
        logging.info("fd and ai matrix")
        fd_mat, ai_mat = pre_fdai_mat_eigen_glm(y, xmat, leg_tp, kin_eigen_val, cov_add, cov_per, cov_res, var_com,
                                            var_ind, kin_dct)
        logging.info("EM matrix")
        em_mat = pre_em_mat_eigen(cov_dim, cov_add, cov_per, ran_df, var_com, num_record)
        gamma = -em_weight_step
        while gamma < 1.0:
            gamma = gamma + em_weight_step
            if gamma >= 1.0:
                gamma = 1.0
            wemai_mat = (1 - gamma) * ai_mat + gamma * em_mat
            delta = np.dot(linalg.inv(wemai_mat), fd_mat)
            var_com_update = var_com + delta
            cov = pre_cov_mat_eigen(cov_dim, var_com_update)
            if cov is not None:
                logging.info('EM weight value: ' + str(gamma))
                break
        logging.info('Updated variances: ' + ' '.join(np.array(var_com_update, dtype=str)))
        cov_add, cov_per, cov_res = cov[:]
        # Convergence criteria
        cc_par_val = np.sum(pow(delta, 2)) / np.sum(pow(var_com_update, 2))
        cc_par_val = np.sqrt(cc_par_val)
        cc_gra_val = np.sqrt(np.sum(pow(fd_mat, 2))) / len(var_com)
        var_com = var_com_update.copy()
        logging.info("Change in parameters, Norm of gradient vector: {} {}".format(cc_par_val, cc_gra_val))
        if cc_par_val < cc_par and cc_gra_val < cc_gra:
            break
    if cc_par_val < cc_par and cc_gra_val < cc_gra:
        logging.info("Variances Converged")
    else:
        logging.info("Variances Not Converged")
    var_pd = {'vari': np.array(var_ind[:, 0]) + 1,
              "varij": np.array(var_ind[:, 1]) + 1,
              "varik": np.array(var_ind[:, 2]) + 1,
              "var_val": var_com}
    var_pd = pd.DataFrame(var_pd, columns=['vari', "varij", "varik", "var_val"])
    return var_pd

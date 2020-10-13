import os
import logging
from scipy import linalg
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import gc

from .common import *
from .balance_emai import *


def balance_varcom(data_file, id, tpoint, trait, kin_file, tfix=None, fix=None, forder=3, rorder=3, na_method='omit',
                   init=None, maxiter=100, cc_par=1.0e-8, cc_gra=1.0e6, em_weight_step=0.001,
                   prefix_outfile='balance_varcom'):
    """
    Estimate variance parameters for balanced longitudinal data.
    :param data_file: the data file. The first row is the variate names whose first initial position is alphabetical.
    For the class variates, the first letter must be capital; for the covariates (continuous variates), the first letter
    must be lowercase.
    :param id: A class variate name which indicates the individual id column in the data file.
    :param tpoint: A list of corresponding time points for phenotypic values.
    :param trait: A list indicating the columns for recorded phenotypic values. The column index starts from 0 in the
    data file.
    :param kin_file: the file for genomic relationship matrix. This file can be produced by
    gmat.gmatrix.agmat function using agmat(bed_file, inv=True, small_val=0.001, out_fmt='id_id_val')
    :param tfix: A class variate name for the time varied fixed effect. Default value is None. The value must be None
    in the current version.
    :param fix: Expression for the time independent fixed effect. Default value is None. The value must be None
    in the current version.
    :param forder: the order of Legendre polynomials for the time varied fixed effect. The default value is 3.
    :param rorder: the order of Legendre polynomials for time varied random effects (additive genetic effects and
    permanent environment effects). The default value is 3.
    :param na_method: The method to deal with missing values. The default value is 'omit'. 'omit' method will delete the
    row with missing values. 'include' method will fill the missing values with the adjacent values.
    :param init: the initial values for the variance parameters. The default value is None.
    :param maxiter: the maximum number of iteration. Default is 100.
    :param cc_par: Convergence criteria for the changed variance parameters. Default is 1.0e-8.
    :param cc_gra: Convergence criteria for the norm of gradient vector. Default is 1.0e6.
    :param em_weight_step: the step of the em weight. Default is 0.001.
    :param prefix_outfile: the prefix for the output file. Default is 'balance_varcom'.
    :return: the estimated variance parameters.
    """
    logging.info('########################################################################')
    logging.info('###Prepare the data for unbalanced longitudinal variances estimation.###')
    logging.info('########################################################################')
    logging.info('***Read the data file***')
    logging.info('Data file: ' + data_file)
    data_df = pd.read_csv(data_file, sep='\s+', header=0)
    logging.info('NA method: ' + na_method)
    if na_method == 'omit':
        data_df = data_df.dropna()
    elif na_method == 'include':
        data_df = data_df.fillna(method='ffill')
        data_df = data_df.fillna(method='bfill')
    else:
        logging.info('na_method does not exist: ' + na_method)
        exit()
    col_names = data_df.columns
    logging.info('The column names of data file: ' + ' '.join(list(col_names)))
    logging.info('Note: Variates beginning with a capital letter is converted into factors.')
    class_vec = []
    for val in col_names:
        if not val[0].isalpha():
            logging.info("The first character of columns names must be alphabet!")
            exit()
        if val[0] == val.capitalize()[0]:
            class_vec.append(val)
            data_df[val] = data_df[val].astype('str')
        else:
            try:
                data_df[val] = data_df[val].astype('float')
            except Exception as e:
                logging.info(e)
                logging.info(val + " column may contain string, please check!")
                exit()
    logging.info('Individual column: ' + id)
    if id not in col_names:
        logging.info(id + ' is not in the data file, please check!')
        exit()
    if id not in class_vec:
        logging.info('The initial letter of {} should be capital'.format(id))
        exit()
    id_in_data_lst = list(data_df[id])
    id_in_data = set(id_in_data_lst)
    if len(id_in_data_lst) != len(id_in_data):
        logging.info('Duplicated ids exit in the data file, please check!')
        exit()
    logging.info('Trait column: ' + ' '.join(np.array(trait, dtype=str)))
    logging.info('Trait column name: ' + ' '.join(list(col_names[trait])))
    if len(set(col_names[trait]) & set(class_vec)) != 0:
        logging.info('Phenotype should not be defined as class variable, please check!')
        exit()
    logging.info('Code factor variables of the data file: ' + ' '.join(list(class_vec)))
    code_dct = {}
    for val in class_vec:
        code_val = 0
        code_dct[val] = {}
        col_vec = []
        for i in range(data_df.shape[0]):
            if data_df[val][i] not in code_dct[val]:
                code_val += 1
                code_dct[val][data_df[val][i]] = str(code_val)
            col_vec.append(code_dct[val][data_df[val][i]])
        data_df[val] = np.array(col_vec)
        data_df[val] = data_df[val].astype('int')
    logging.info('***Read the kinship matrix***')
    with open(kin_file) as fin:
        row = []
        col = []
        kin = []
        id_in_kin = set()
        for line in fin:
            arr = line.split()
            if arr[0] not in code_dct[id] or arr[1] not in code_dct[id]:
                continue
            id_in_kin.add(arr[0])
            id_in_kin.add(arr[1])
            row.append(int(code_dct[id][arr[0]]))
            col.append(int(code_dct[id][arr[1]]))
            kin.append(float(arr[2]))
        kin = csr_matrix((np.array(kin), (np.array(row) - 1, np.array(col) - 1))).toarray()
        kin = np.add(kin, kin.T)
        np.fill_diagonal(kin, 0.5 * np.diag(kin))
        del row, col
        gc.collect()
    logging.info('***Eigen decomposition of kinship matrix***')
    id_not_in_kin = list(id_in_data - id_in_kin)
    if len(id_not_in_kin) != 0:
        logging.info('The ID: {} in the data file is not in the kinship file, please remove these IDs!'.format(' '.join(id_not_in_kin)))
        exit()
    kin_eigen_val, kin_eigen_vec = linalg.eigh(kin)
    logging.info('***Build the design matrix for fixed effect***')
    leg_fix = leg(np.array(tpoint), forder)
    leg_fix = np.array(leg_fix).reshape(forder + 1, 1, len(tpoint))
    leg_fix = np.concatenate([leg_fix] * data_df.shape[0], axis=1)
    xmat_t = leg_fix.copy()
    if tfix is not None:
        logging.info('The parameter tfix should be None in current version.')
        exit()
    xmat_t = np.matmul(np.array([kin_eigen_vec.T]), xmat_t)
    xmat_t = xmat_t.transpose(1, 2, 0)
    if fix is not None:
        logging.info('The parameter fix should be None in current version')
        exit()
    logging.info('***Initial values for variances and rotate the phenotypic values')
    leg_tp = leg(np.array(tpoint), rorder)
    leg_tp = np.concatenate(leg_tp, axis=1)
    y = np.array(data_df.iloc[:, trait])
    cov_dim = leg_tp.shape[1]
    y_var = np.var(y) / (cov_dim*2 + 1)
    var_com = []
    cov_var = np.diag([y_var]*cov_dim)
    var_com.extend(cov_var[np.tril_indices_from(cov_var)])
    var_com.extend(cov_var[np.tril_indices_from(cov_var)])
    var_com.append(y_var)
    if init is None:
        var_com = np.array(var_com)
    else:
        if len(var_com) != len(init):
            logging.info('The length of initial variances should be {}'.format(len(var_com)))
            exit()
        else:
            var_com = np.array(init)
    y = np.dot(kin_eigen_vec.T, y)
    y = y.reshape(data_df.shape[0], len(tpoint), 1)
    logging.info('##########################################################')
    logging.info('###Start the balanced longitudinal variances estimation###')
    logging.info('##########################################################')
    res = balance_emai(y, xmat_t, leg_tp, kin_eigen_val, init=var_com, maxiter=maxiter, cc_par=cc_par,
                       cc_gra=cc_gra, em_weight_step=em_weight_step)
    var_file = prefix_outfile + '.var'
    res.to_csv(var_file, sep=' ', index=False)
    return res

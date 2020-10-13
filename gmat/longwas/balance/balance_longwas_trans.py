import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from gmat.process_plink.process_plink import read_plink, impute_geno
import pandas as pd
import datetime
from scipy.stats import chi2
import gc
import logging
from tqdm import tqdm
from functools import reduce

from .common import *


def balance_longwas_trans(data_file, id, tpoint, trait, kin_file, bed_file, var_com, snp_lst=None, tfix=None, fix=None,
            forder=3, rorder=3, na_method='omit', prefix_outfile='balance_longwas_trans'):
    """
    Longitudinal GWAS for balanced data.
    :param data_file:the data file. The first row is the variate names whose first initial position is alphabetical.
    For the class variates, the first letter must be capital; for the covariates (continuous variates), the first letter
    must be lowercase.
    :param id: A class variate name which indicates the individual id column in the data file.
    :param tpoint: A list of corresponding time points for phenotypic values.
    :param trait: A list indicating the columns for recorded phenotypic values. The column index starts from 0 in the
    data file.
    :param kin_file: the file for genomic relationship matrix. This file can be produced by
    gmat.gmatrix.agmat function using agmat(bed_file, inv=True, small_val=0.001, out_fmt='id_id_val')
    :param bed_file: the plink binary file
    :param var_com: variances parameters from the balance_varcom function.
    :param snp_lst: A list of snp to test. Default is None.
    :param tfix: A class variate name for the time varied fixed effect. Default value is None. The value must be None
    in the current version.
    :param fix: Expression for the time independent fixed effect. Default value is None. The value must be None
    in the current version.
    :param forder: the order of Legendre polynomials for the time varied fixed effect. The default value is 3.
    :param rorder: the order of Legendre polynomials for time varied random effects (additive genetic effects and
    permanent environment effects). The default value is 3.
    :param na_method: The method to deal with missing values. The default value is 'omit'. 'omit' method will delete the
    row with missing values. 'include' method will fill the missing values with the adjacent values.
    :param prefix_outfile: the prefix for the output file. Default is 'balance_longwas_trans'.
    :return: A pandas dataframe of GWAS results.
    """
    logging.info('################################')
    logging.info('###Prepare the related matrix###')
    logging.info('################################')
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
                logging.info(val + ": may contain string, please check!")
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
    xmat = xmat_t.transpose(1, 2, 0)
    if fix is not None:
        logging.info('The parameter fix should be None in current version')
        exit()
    # T matrix for random effect
    leg_tp = leg(np.array(tpoint), rorder)
    leg_tp = np.concatenate(leg_tp, axis=1)
    y = np.array(data_df.iloc[:, trait])
    y = np.dot(kin_eigen_vec.T, y)
    y = y.reshape(data_df.shape[0], len(tpoint), 1)
    if var_com.shape[0] != rorder*(rorder+1) + 2*(rorder+1) + 1:
        logging.info('Variances do not match the data, please check')
        exit()
    logging.info('***Read the snp data***')
    snp_mat = read_plink(bed_file)
    if np.any(np.isnan(snp_mat)):
        logging.info('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    num_id = snp_mat.shape[0]
    num_snp = snp_mat.shape[1]
    logging.info("There are {:d} individuals and {:d} SNPs.".format(num_id, num_snp))
    fam_df = pd.read_csv(bed_file + '.fam', sep='\s+', header=None)
    id_in_geno = list(np.array(fam_df.iloc[:, 1], dtype=str))
    if len(set(id_in_data_lst) - set(id_in_geno)) != 0:
        logging.info(' '.join(list(set(id_in_data_lst) - set(id_in_geno))) + ' in the data file is not in the snp file!')
        exit()
    # snp list
    if snp_lst is None:
        snp_lst = range(num_snp)
    else:
        try:
            snp_lst = np.array(snp_lst, dtype=int)
        except Exception as e:
            logging.info(e)
            logging.info('The snp list value should be int')
            exit()
    snp_lst = list(snp_lst)
    id_in_data_index = []
    for i in id_in_data_lst:
        id_in_data_index.append(id_in_geno.index(i))
    snp_mat = snp_mat[id_in_data_index, :]
    snp_mat = snp_mat[:, snp_lst]
    freq = np.sum(snp_mat, axis=0) / (2 * num_id)
    freq.shape = (1, len(freq))
    snp_mat = snp_mat - 2 * freq
    snp_mat = np.dot(kin_eigen_vec.T, snp_mat)
    logging.info('***Prepare the variance structure***')
    add_cov = var_com.loc[var_com.loc[:, 'vari']==1, :]
    row = np.array(add_cov['varij']) - 1
    col = np.array(add_cov['varik']) - 1
    val = add_cov['var_val']
    add_cov = csr_matrix((val, (row, col))).toarray()
    add_cov = add_cov + np.tril(add_cov, k=-1).T
    per_cov = var_com.loc[var_com.loc[:, 'vari'] == 2, :]
    row = np.array(per_cov['varij']) - 1
    col = np.array(per_cov['varik']) - 1
    val = per_cov['var_val']
    per_cov = csr_matrix((val, (row, col))).toarray()
    per_cov = per_cov + np.tril(per_cov, k=-1).T
    res_var = np.array(var_com['var_val'])[-1]
    vinv = np.multiply(kin_eigen_val.reshape(len(kin_eigen_val), 1, 1), tri_matT(leg_tp, add_cov)) + \
        tri_matT(leg_tp, per_cov) + np.diag([np.sum(res_var)] * leg_tp.shape[0])
    vinv = np.linalg.inv(vinv)
    vx = np.matmul(vinv, xmat)
    xvx = np.matmul(xmat.transpose(0, 2, 1), vx)
    xvx = reduce(np.add, xvx)
    xvx = np.linalg.inv(xvx)
    xvy = np.matmul(xmat.transpose(0, 2, 1), np.matmul(vinv, y))
    xvy = reduce(np.add, xvy)
    y_xb = y - np.matmul(xmat, np.dot(xvx, xvy))
    py = np.matmul(vinv, y_xb)
    logging.info('########################################################################')
    logging.info('###Start the linear transformation longitudinal GWAS for balance data###')
    logging.info('########################################################################')
    leg_tpoint_mat = leg_mt(np.array(tpoint), max(tpoint), min(tpoint), forder)
    leg_tpoint_accum = np.sum(leg_tpoint_mat, axis=0)
    chi_df = leg_tp.shape[1]
    eff_vec = []
    chi_vec = []
    p_vec = []
    p_min_vec = []
    p_accum_vec = []
    temp_gt = np.dot(add_cov, leg_tp.T)
    for i in tqdm(range(snp_mat.shape[1])):
        snpi = snp_mat[:, i].reshape(num_id, 1, 1)
        snpi = np.multiply(snpi, temp_gt)
        snp_eff = np.matmul(snpi, py)
        snp_eff = reduce(np.add, snp_eff)
        snp_cov1 = reduce(np.matmul, [snpi, vinv, snpi.transpose(0, 2, 1)])
        snp_cov1 = reduce(np.add, snp_cov1)
        snp_cov2 = np.matmul(snpi, vx)
        snp_cov2 = reduce(np.add, snp_cov2)
        snp_cov2 = reduce(np.dot, [snp_cov2, xvx, snp_cov2.T])
        snp_cov = snp_cov1 - snp_cov2
        chi_val = np.sum(np.dot(np.dot(snp_eff.T, np.linalg.inv(snp_cov)), snp_eff))
        p_val = chi2.sf(chi_val, chi_df)
        eff_vec.append(snp_eff)
        chi_vec.append(chi_val)
        p_vec.append(p_val)
        p_tpoint_vec = []
        for k in range(leg_tpoint_mat.shape[0]):
            eff_tpoint = np.sum(np.dot(leg_tpoint_mat[k, :], snp_eff))
            eff_var_tpoint = np.sum(np.dot(leg_tpoint_mat[k, :], np.dot(snp_cov, leg_tpoint_mat[k, :])))
            chi_tpoint = eff_tpoint * eff_tpoint / eff_var_tpoint
            p_tpoint = chi2.sf(chi_tpoint, 1)
            p_tpoint_vec.append(p_tpoint)
        p_min_vec.append(min(p_tpoint_vec))
        eff_accum = np.sum(np.dot(leg_tpoint_accum, snp_eff))
        eff_var_accum = np.sum(np.dot(leg_tpoint_accum, np.dot(snp_cov, leg_tpoint_accum)))
        chi_accum = eff_accum * eff_accum / eff_var_accum
        p_accum = chi2.sf(chi_accum, 1)
        p_accum_vec.append(p_accum)
    logging.info('Finish association analysis')
    logging.info('***Output***')
    snp_info_file = bed_file + '.bim'
    snp_info = pd.read_csv(snp_info_file, sep='\s+', header=None)
    res_df = snp_info.iloc[snp_lst, [0, 1, 3, 4, 5]]
    res_df.columns = ['chro', 'snp_ID', 'pos', 'allele1', 'allele2']
    res_df.loc[:, 'order'] = snp_lst
    res_df = res_df.iloc[:, [5, 0, 1, 2, 3, 4]]
    eff_vec = np.array(eff_vec)
    for i in range(eff_vec.shape[1]):
        col_ind = 'eff' + str(i)
        res_df.loc[:, col_ind] = eff_vec[:, i]
    res_df.loc[:, 'chi_val'] = chi_vec
    res_df.loc[:, 'p_val'] = p_vec
    res_df.loc[:, 'p_min'] = p_min_vec
    res_df.loc[:, 'p_accum'] = p_accum_vec
    out_file = prefix_outfile + '.res'
    res_df.to_csv(out_file, sep=' ', index=False)
    return res_df

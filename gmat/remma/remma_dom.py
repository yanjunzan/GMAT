import numpy as np
from scipy import linalg
import pandas as pd
from functools import reduce
import gc
import time
import logging
import sys
from scipy.stats import chi2

from gmat.process_plink.process_plink import read_plink, impute_geno
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat


def _remma_dom(y, xmat, zmat, gmat_lst, var_com, bed_file, out_file='remma_dom'):
    """
    Dominance test by random SNP-BLUP model.
    :param y: phenotypic vector
    :param xmat: Designed matrix for fixed effect
    :param zmat: csr sparse matrix. Designed matrix for random effect.
    :param gmat_lst: A list for relationship matrix
    :param var_com: A list of estimated variances. var_com[0]: additive variances; var_com[1]: dominance variances
    :param bed_file: the prefix for plink binary file
    :param out_file: The output file. Default is 'remma_dom'.
    :return: pandas data frame for results.
    """
    logging.info("Calculate the phenotypic covariance matrix and inversion")
    y = np.array(y).reshape(-1, 1)
    n = y.shape[0]
    xmat = np.array(xmat).reshape(n, -1)
    vmat = np.diag([var_com[-1]] * n)
    for val in range(len(gmat_lst)):
        vmat += zmat.dot((zmat.dot(gmat_lst[val])).T) * var_com[val]
    # del gmat_lst
    # gc.collect()
    vmat_inv = linalg.inv(vmat)
    logging.info("Calculate P matrix")
    vxmat = np.dot(vmat_inv, xmat)
    xvxmat = np.dot(xmat.T, vxmat)
    xvxmat = linalg.inv(xvxmat)
    pmat = reduce(np.dot, [vxmat, xvxmat, vxmat.T])
    pmat = vmat_inv - pmat
    pymat = zmat.T.dot(np.dot(pmat, y))
    # pvpmat = reduce(np.dot, [pmat, vmat, pmat])
    pvpmat = zmat.T.dot((zmat.T.dot(pmat)).T)  # pvp = p
    del vmat, vmat_inv, pmat
    gc.collect()
    logging.info("Read the SNP")
    snp_mat = read_plink(bed_file)
    num_id, num_snp = snp_mat.shape
    if np.any(np.isnan(snp_mat)):
        logging.warning('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    freq = np.sum(snp_mat, axis=0) / (2 * num_id)
    freq.shape = (1, num_snp)
    scale_vec = 2 * freq * (1 - freq)
    scale = np.sum(scale_vec * (1 - scale_vec))
    logging.info('The scaled factor is: {:.3f}'.format(scale))
    snp_mat[snp_mat > 1.5] = 0.0
    snp_mat = snp_mat - scale_vec
    eff_vec = np.dot(snp_mat.T, pymat)[:, -1] * var_com[1] / scale
    var_vec = np.sum(snp_mat * np.dot(pvpmat, snp_mat), axis=0) * var_com[1] * var_com[1]/(scale*scale)
    eff_vec_to_fixed = eff_vec * var_com[1] / (var_vec * scale)
    chi_vec = eff_vec*eff_vec/var_vec
    p_vec = chi2.sf(chi_vec, 1)
    snp_info_file = bed_file + '.bim'
    snp_info = pd.read_csv(snp_info_file, sep='\s+', header=None)
    res_df = snp_info.iloc[:, [0, 1, 3, 4, 5]]
    res_df.columns = ['chro', 'snp_ID', 'pos', 'allele1', 'allele2']
    res_df.loc[:, 'eff_val'] = eff_vec
    res_df.loc[:, 'chi_val'] = chi_vec
    res_df.loc[:, 'eff_val_to_fixed'] = eff_vec_to_fixed
    res_df.loc[:, 'p_val'] = p_vec
    try:
        res_df.to_csv(out_file, index=False, header=True, sep=' ')
    except Exception as e:
        logging.error(e)
        sys.exit()
    return res_df


def remma_dom(pheno_file, bed_file, gmat_lst, var_com, out_file='remma_dom'):
    """
    Dominance test by random SNP-BLUP model.
    :param pheno_file: phenotypic file. The fist two columns are family id, individual id which are same as plink *.fam
    file. The third column is always ones for population mean. The last column is phenotypic values. The ohter covariate
    can be added between columns for population mean and phenotypic values.
    :param bed_file: the prefix for binary file
    :param gmat_lst: a list of genomic relationship matrixes.
    :param var_com: Estimated variances
    :param out_file: output file. default value is 'remma_dom'.
    :return: pandas data frame for results.
    """
    y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)
    res = _remma_dom(y, xmat, zmat, gmat_lst, var_com, bed_file, out_file=out_file)
    return res

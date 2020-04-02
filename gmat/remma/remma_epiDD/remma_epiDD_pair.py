import numpy as np
from scipy import linalg
import pandas as pd
from functools import reduce
import gc
import time
import logging
from tqdm import tqdm
import sys
from scipy.stats import chi2

from gmat.process_plink.process_plink import read_plink, impute_geno


def remma_epiDD_pair(y, xmat, zmat, gmat_lst, var_com, bed_file, snp_pair_file, max_test_pair=50000, p_cut=1.0e-4, out_file='remma_epiDD_pair'):
    """
    Given a SNP pair file, perform dominance by dominance epistasis test by random SNP-BLUP model.
    :param y: phenotypic vector
    :param xmat: Designed matrix for fixed effect
    :param zmat: csr sparse matrix. Designed matrix for random effect.
    :param gmat_lst: A list for relationship matrix
    :param var_com: Estimated variances
    :param bed_file: the prefix for plink binary file
    :param snp_pair_file: a file containing index for SNP pairs. The program only reads the first two columns and test
    SNP pairs row by row. The max value is num_snp - 1, and the min value is 0.
    :param max_test_pair: The max number of SNP pairs stored in memory. Default value is 50000.
    :param p_cut: put cut value. default value is 0.0001.
    :param out_file: output file. default value is 'remma_epiDD_pair'.
    :return: 0
    """
    logging.info("Calculate the phenotypic covariance matrix and inversion")
    y = np.array(y).reshape(-1, 1)
    n = y.shape[0]
    xmat = np.array(xmat).reshape(n, -1)
    vmat = np.diag([var_com[-1]] * n)
    for val in range(len(gmat_lst)):
        vmat += zmat.dot((zmat.dot(gmat_lst[val])).T) * var_com[val]
    del gmat_lst
    gc.collect()
    vmat_inv = np.linalg.inv(vmat)
    logging.info("Calculate P matrix")
    vxmat = np.dot(vmat_inv, xmat)
    xvxmat = np.dot(xmat.T, vxmat)
    xvxmat = np.linalg.inv(xvxmat)
    pmat = reduce(np.dot, [vxmat, xvxmat, vxmat.T])
    pmat = vmat_inv - pmat
    pymat = zmat.T.dot(np.dot(pmat, y))
    # pvpmat = reduce(np.dot, [pmat, vmat, pmat])  # pvp = p
    pvpmat = zmat.T.dot((zmat.T.dot(pmat)).T)
    del vmat, vmat_inv, pmat
    gc.collect()
    logging.info("Read the SNP")
    snp_mat = read_plink(bed_file)
    num_id, num_snp = snp_mat.shape
    if np.any(np.isnan(snp_mat)):
        logging.warning('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    freq = np.sum(snp_mat, axis=0) / (2 * num_id)
    freq.shape = (1, -1)
    scale_vec = 2 * freq * (1 - freq)
    scale = np.sum(scale_vec * (1 - scale_vec))
    logging.info('The scaled factor is: {:.3f}'.format(scale))
    snp_mat[snp_mat > 1.5] = 0.0  # 2替换为0, 变为0、1、0编码
    snp_mat = snp_mat - scale_vec
    logging.info("Test")
    np.savetxt(out_file, ['snp_0 snp_1 eff var chi p'], fmt='%s')
    clock_t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    ipart = -1
    while True:
        ipart += 1
        skiprows = 1 + ipart * max_test_pair
        try:
            snp_pair = pd.read_csv(snp_pair_file, header=None, sep='\s+', skiprows=skiprows, nrows=max_test_pair)
        except Exception as e:
            logging.info(e)
            break
        snp_pair = np.array(snp_pair.iloc[:, 0:2], dtype=np.int)
        if np.max(snp_pair) > num_snp - 1 or np.min(snp_pair) < 0:
            logging.error('snp_pair is out of range!')
            sys.exit()
        epi_mat = snp_mat[:, snp_pair[:, 0]] * snp_mat[:, snp_pair[:, 1]]
        eff_vec = np.dot(epi_mat.T, pymat)
        var_vec = np.sum(epi_mat * np.dot(pvpmat, epi_mat), axis=0)
        var_vec = var_vec.reshape(-1, 1)
        chi_vec = eff_vec * eff_vec / var_vec
        p_vec = chi2.sf(chi_vec, 1)
        res = pd.DataFrame(
            {0: snp_pair[:, 0], 1: snp_pair[:, 1], 2: eff_vec[:, -1], 3: var_vec[:, -1], 4: chi_vec[:, -1],
             5: p_vec[:, -1]})
        res = res[res[5] < p_cut]
        res.to_csv(out_file, sep=' ', header=False, index=False, mode='a')
    clock_t1 = time.perf_counter()
    cpu_t1 = time.process_time()
    logging.info("Running time: Clock time, {:.5f} sec; CPU time, {:.5f} sec.".format(clock_t1 - clock_t0, cpu_t1 - cpu_t0))
    return 0

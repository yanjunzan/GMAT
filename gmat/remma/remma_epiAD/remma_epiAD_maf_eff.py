import numpy as np
from scipy import linalg
import pandas as pd
from functools import reduce
import gc
import time
import logging
from tqdm import tqdm
import sys
import os
from scipy.stats import chi2

from gmat.process_plink.process_plink import read_plink, impute_geno
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat


from _cremma_epi_eff_cpu import ffi, lib


def _remma_epiAD_maf_eff(y, xmat, zmat, gmat_lst, var_com, bed_file, snp_lst_0=None, freqA=None, freqD=None,
                         freq_deno=None, p_cut=1.0e-5, out_file='epiAD_maf_eff'):
    """
    Estimate additive by additive epistasis effects by random SNP-BLUP model.
    :param y: phenotypic vector
    :param xmat: Designed matrix for fixed effect
    :param zmat: csr sparse matrix. Designed matrix for random effect.
    :param gmat_lst: A list for relationship matrix
    :param var_com: Estimated variances
    :param bed_file: the prefix for plink binary file
    :param snp_lst_0: the first SNP list for the SNP pairs. the min value is 0 and the max value is num_snp-2. The
    default value is None, which means list [0, num_snp-1)
    :param var_app: the approximate variances for estimated SNP effects.
    :param p_cut: put cut value. default value is 1.0e-5.
    :param out_file: output file. default value is 'epiAA_eff'.
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
    del vmat, vmat_inv, pmat
    gc.collect()
    num_snp = pd.read_csv(bed_file+'.bim', header=None).shape[0]
    num_id = pd.read_csv(bed_file+'.fam', header=None).shape[0]
    if snp_lst_0 is None:
        snp_lst_0 = range(num_snp - 1)
    else:
        if max(snp_lst_0) >= num_snp - 1 or min(snp_lst_0) < 0:
            logging.error('snp_lst_0 is out of range!')
            sys.exit()
    if freqA is None:
        freqA = np.zeros((num_snp,), dtype=np.longlong)
    if freqD is None:
        freqD = np.zeros((num_snp,), dtype=np.longlong)
    if freq_deno is None:
        freq_deno = np.ones(111)
    logging.info("Convert python variates to C type")
    pbed_file = ffi.new("char[]", bed_file.encode('ascii'))
    pnum_id = ffi.cast("long long", num_id)
    pnum_snp = ffi.cast("long long", num_snp)
    snp_lst_0 = np.array(list(snp_lst_0), dtype=np.longlong)
    psnp_lst_0 = ffi.cast("long long *", snp_lst_0.ctypes.data)
    # psnp_lst_0 = ffi.cast("long long *", ffi.from_buffer(snp_lst_0))
    plen_snp_lst_0 = ffi.cast("long long", len(snp_lst_0))
    pfreqA = ffi.cast("long long *", freqA.ctypes.data)
    pfreqD = ffi.cast("long long *", freqD.ctypes.data)
    ppymat = ffi.cast("double *", pymat.ctypes.data)
    chi_cut = chi2.isf(p_cut, 1)
    eff_cut = np.array(np.sqrt(chi_cut * freq_deno))
    peff_cut = ffi.cast("double *", eff_cut.ctypes.data)
    # peff_cut = ffi.cast("double", 1.0)
    temp_file = out_file + '.temp'
    pout_file = ffi.new("char[]", temp_file.encode('ascii'))
    logging.info('Test')
    clock_t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    lib.remma_epiAD_maf_eff_cpu(pbed_file, pnum_id, pnum_snp, psnp_lst_0, plen_snp_lst_0, ppymat, pfreqA, pfreqD, peff_cut, pout_file)
    clock_t1 = time.perf_counter()
    cpu_t1 = time.process_time()
    logging.info("Running time: Clock time, {:.5f} sec; CPU time, {:.5f} sec.".format(clock_t1 - clock_t0, cpu_t1 - cpu_t0))
    logging.info('Add the approximate P values')
    with open(temp_file) as fin, open(out_file, 'w') as fout:
        head_line = fin.readline()
        head_line = head_line.strip()
        head_line += ' chi_app p_app\n'
        fout.write(head_line)
        for line in fin:
            arr = line.split()
            chi_app = float(arr[-1]) * float(arr[-1]) / freq_deno[freqA[int(arr[0])] * 10 + freqD[int(arr[1])]]
            p_app = chi2.sf(chi_app, 1)
            fout.write(' '.join(arr + [str(chi_app), str(p_app)]) + '\n')
    os.remove(temp_file)
    return 0


def remma_epiAD_maf_eff(pheno_file, bed_file, gmat_lst, var_com, snp_lst_0=None, freqA=None, freqD=None, freq_deno=None, p_cut=1.0e-5, out_file='epiAA_maf_eff'):
    """
    Estimate additive by additive epistasis effects by random SNP-BLUP model.
    :param pheno_file: phenotypic file. The fist two columns are family id, individual id which are same as plink *.fam
    file. The third column is always ones for population mean. The last column is phenotypic values. The ohter covariate
    can be added between columns for population mean and phenotypic values.
    :param bed_file: the prefix for binary file
    :param gmat_lst: A list for relationship matrix
    :param var_com: Estimated variances
    :param snp_lst_0: the first SNP list for the SNP pairs. the min value is 0 and the max value is num_snp-2. The
    default value is None, which means list [0, num_snp-1)
    :param var_app: the approximate variances for estimated SNP effects.
    :param p_cut: put cut value. default value is 1.0e-5.
    :param out_file: output file. default value is 'epiAA_eff'.
    :return: 0
    """
    y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)
    res = _remma_epiAD_maf_eff(y, xmat, zmat, gmat_lst, var_com, bed_file,
                           snp_lst_0=snp_lst_0, freqA=freqA, freqD=freqD, freq_deno=freq_deno, p_cut=p_cut, out_file=out_file)
    return res


def _remma_epiAD_maf_eff_parallel(y, xmat, zmat, gmat_lst, var_com, bed_file, parallel,
                                freqA=None, freqD=None, freq_deno=None, p_cut=1.0e-5, out_file='epiAA_maf_eff_parallel'):
    """
    Parallel version. Additive by additive epistasis test by random SNP-BLUP model.
    :param y: phenotypic vector
    :param xmat: Designed matrix for fixed effect
    :param zmat: csr sparse matrix. Designed matrix for random effect.
    :param gmat_lst: A list for relationship matrix
    :param var_com: Estimated variances
    :param bed_file: the prefix for plink binary file
    :param parallel: A list containing two integers. The first integer is the number of parts to parallel. The second
    integer is the part to run. For example, parallel = [3, 1], parallel = [3, 2] and parallel = [3, 3] mean to divide
    total number of tests into three parts and run parallelly.
    :param var_app: the approximate variances for estimated SNP effects.
    :param p_cut: put cut value. default value is 1.0e-5.
    :param out_file: output file. default value is 'remma_epiAA_eff_parallel'.
    :return: 0
    """
    logging.info("Parallel: " + str(parallel[0]) + ', ' + str(parallel[1]))
    bim_df = pd.read_csv(bed_file + '.bim', header=None)
    num_snp = bim_df.shape[0]
    num_snp_part = int(num_snp/(2*parallel[0]))
    snp_pos_0 = (parallel[1]-1) * num_snp_part
    snp_pos_1 = parallel[1] * num_snp_part
    snp_pos_2 = (2*parallel[0] - parallel[1]) * num_snp_part
    snp_pos_3 = (2*parallel[0] - parallel[1] + 1) * num_snp_part
    if parallel[1] == 1:
        snp_pos_3 = num_snp - 1
    logging.info('SNP position point: ' +
                 ','.join(list(np.array([snp_pos_0, snp_pos_1, snp_pos_2, snp_pos_3], dtype=str))))
    snp_list_0 = list(range(snp_pos_0, snp_pos_1)) + list(range(snp_pos_2, snp_pos_3))
    res = _remma_epiAD_maf_eff(y, xmat, zmat, gmat_lst, var_com, bed_file, snp_lst_0=snp_list_0, freqA=freqA, freqD=freqD,
                               freq_deno=freq_deno, p_cut=p_cut, out_file=out_file + '.' + str(parallel[1]))
    return res


def remma_epiAD_maf_eff_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel,
                                   freqA=None, freqD=None, freq_deno=None,  p_cut=1.0e-5, out_file='epiAA_maf_eff_parallel'):
    """
    Parallel version. Additive by additive epistasis test by random SNP-BLUP model.
    :param pheno_file: phenotypic file. The fist two columns are family id, individual id which are same as plink *.fam
    file. The third column is always ones for population mean. The last column is phenotypic values. The ohter covariate
    can be added between columns for population mean and phenotypic values.
    :param bed_file: the prefix for binary file
    :param gmat_lst: A list for relationship matrix
    :param var_com: Estimated variances
    :param parallel: A list containing two integers. The first integer is the number of parts to parallel. The second
    integer is the part to run. For example, parallel = [3, 1], parallel = [3, 2] and parallel = [3, 3] mean to divide
    :param var_app: the approximate variances for estimated SNP effects.
    :param p_cut: put cut value. default value is 1.0e-5.
    :param out_file: output file. default value is 'epiAA_maf_eff_parallel'.
    :return: 0
    """
    y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)
    res = _remma_epiAD_maf_eff_parallel(y, xmat, zmat, gmat_lst, var_com, bed_file, parallel,
                                   freqA=freqA, freqD=freqD, freq_deno=freq_deno,  p_cut=p_cut, out_file=out_file)
    return res


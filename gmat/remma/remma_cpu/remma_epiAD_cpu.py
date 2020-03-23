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


def remma_epiAD_cpu(y, xmat, zmat, gmat_lst, var_com, bed_file, snp_lst_0=None, p_cut=0.0001, out_file='remma_epiAD_cpu'):
    """
    additive by dominance epistasis test by random SNP-BLUP model.
    :param y: phenotypic vector
    :param xmat: Designed matrix for fixed effect
    :param zmat: csr sparse matrix. Designed matrix for random effect.
    :param gmat_lst: A list for relationship matrix
    :param var_com: Estimated variances
    :param bed_file: the prefix for plink binary file
    :param snp_lst_0: the first SNP list for the SNP pairs. the min value is 0 and the max value is num_snp-1. The
    default value is None, which means list [0, num_snp-1]
    :param p_cut: put cut value. default value is 0.0001.
    :param out_file: output file. default value is 'remma_epiAD_cpu'.
    :return:
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
    # pvpmat = reduce(np.dot, [pmat, vmat, pmat])  # pvp = p
    pvpmat = zmat.T.dot((zmat.T.dot(pmat)).T)
    del vmat, vmat_inv, pmat
    gc.collect()
    logging.info("Read the SNP")
    np.savetxt(out_file, ['snp_0 snp_1 eff chi p_val'], fmt='%s')
    snp_mat = read_plink(bed_file)
    num_id, num_snp = snp_mat.shape
    if np.any(np.isnan(snp_mat)):
        logging.warning('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    freq = np.sum(snp_mat, axis=0) / (2 * num_id)
    freq.shape = (1, num_snp)
    snp_matA = snp_mat - 2 * freq
    snp_mat[snp_mat > 1.5] = 0.0  # 2替换为0, 变为0、1、0编码
    snp_matD = snp_mat - 2 * freq * (1 - freq)
    del snp_mat
    gc.collect()
    logging.info('Test')
    if snp_lst_0 is None:
        snp_lst_0 = range(num_snp)
    else:
        if max(snp_lst_0) > num_snp - 1 or min(snp_lst_0) < 0:
            logging.error('snp_lst_0 is out of range!')
            sys.exit()
    snp_lst_1 = np.arange(num_snp)
    clock_t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    for i in tqdm(snp_lst_0):
        epi_mat = snp_matA[:, i:(i+1)] * snp_matD
        eff_vec = np.dot(epi_mat.T, pymat)
        var_vec = np.sum(epi_mat * np.dot(pvpmat, epi_mat), axis=0)
        var_vec = var_vec.reshape(len(var_vec), -1)
        chi_vec = eff_vec * eff_vec / var_vec
        p_vec = chi2.sf(chi_vec, 1)
        res = pd.DataFrame(
            {0: np.array([i]*num_snp), 1: snp_lst_1, 2: eff_vec[:, -1], 3: chi_vec[:, -1],
             4: p_vec[:, -1]})
        res = res[res[4] < p_cut]
        res.to_csv(out_file, sep=' ', header=False, index=False, mode='a')
    clock_t1 = time.perf_counter()
    cpu_t1 = time.process_time()
    logging.info("Running time: Clock time, {:.5f} sec; CPU time, {:.5f} sec.".format(clock_t1 - clock_t0, cpu_t1 - cpu_t0))
    return 0


def remma_epiAD_select_cpu(y, xmat, zmat, gmat_lst, var_com, bed_file, snp_lst_0=None, snp_lst_1=None, p_cut=1.0, out_file='remma_epiAD_select_cpu'):
    """
    加加上位检验
    :param y: 表型
    :param xmat: 固定效应设计矩阵
    :param zmat: 随机效应设计矩阵，csr稀疏矩阵
    :param gmat_lst: 基因组关系矩阵列表
    :param var_com: 方差组分
    :param bed_file: plink文件
    :param snp_lst_0: 互作对第一个SNP列表,列表最小值为0，最大值为num_snp-1
    :param snp_lst_1: 互作对第一个SNP列表,列表最小值为0，最大值为num_snp-1
    :param p_cut: 依据阈值保留的互作对
    :param out_file: 输出文件
    :return:
    """
    logging.info("计算V矩阵及其逆矩阵")
    y = np.array(y).reshape(-1, 1)
    n = y.shape[0]
    xmat = np.array(xmat).reshape(n, -1)
    vmat = np.diag([var_com[-1]] * n)
    for val in range(len(gmat_lst)):
        vmat += zmat.dot((zmat.dot(gmat_lst[val])).T) * var_com[val]
    del gmat_lst
    gc.collect()
    vmat_inv = np.linalg.inv(vmat)
    logging.info("计算P矩阵")
    vxmat = np.dot(vmat_inv, xmat)
    xvxmat = np.dot(xmat.T, vxmat)
    xvxmat = np.linalg.inv(xvxmat)
    pmat = reduce(np.dot, [vxmat, xvxmat, vxmat.T])
    pmat = vmat_inv - pmat
    pymat = zmat.T.dot(np.dot(pmat, y))
    pvpmat = reduce(np.dot, [pmat, vmat, pmat])
    pvpmat = zmat.T.dot((zmat.T.dot(pvpmat)).T)
    del vmat, vmat_inv, pmat
    gc.collect()
    logging.info("读取SNP文件")
    snp_mat = read_plink(bed_file)
    num_id, num_snp = snp_mat.shape
    if np.any(np.isnan(snp_mat)):
        logging.warning('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    freq = np.sum(snp_mat, axis=0) / (2 * num_id)
    freq.shape = (1, num_snp)
    snp_matA = snp_mat - 2 * freq
    snp_mat[snp_mat > 1.5] = 0.0  # 2替换为0, 变为0、1、0编码
    snp_matD = snp_mat - 2 * freq * (1 - freq)
    del snp_mat
    gc.collect()
    logging.info("开始检验")
    if snp_lst_0 is None:
        snp_lst_0 = range(num_snp)
    else:
        if max(snp_lst_0) > num_snp - 1 or min(snp_lst_0) < 0:
            logging.error('snp_lst_0 is out of range!')
            sys.exit()
    if snp_lst_1 is None:
        snp_lst_1 = range(num_snp)
    else:
        if max(snp_lst_1) > num_snp - 1 or min(snp_lst_1) < 0:
            logging.error('snp_lst_1 is out of range!')
            sys.exit()
    snp_lst_0 = list(snp_lst_0)
    snp_lst_1 = list(snp_lst_1)
    chi2_cut = chi2.isf(p_cut, 1)
    clock_t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    res_lst = []
    for i in snp_lst_0:
        snp_lst_11 = snp_lst_1[:]
        try:
            snp_lst_11.remove(i)
        except Exception as e:
            del e
        epi_mat = snp_matA[:, i:(i + 1)] * snp_matD[:, snp_lst_11]
        eff_vec = np.dot(epi_mat.T, pymat)
        var_vec = np.sum(epi_mat * np.dot(pvpmat, epi_mat), axis=0)
        var_vec = var_vec.reshape(-1, 1)
        chi_vec = eff_vec * eff_vec / var_vec
        res = np.concatenate([np.array([i] * len(snp_lst_11)).reshape(-1, 1), np.array(snp_lst_11).reshape(-1, 1),
                              eff_vec, var_vec, chi_vec], axis=1)
        res_lst.append(res[res[:, -1] > chi2_cut, :])
    clock_t1 = time.perf_counter()
    cpu_t1 = time.process_time()
    logging.info("Running time: Clock time, {:.5f} sec; CPU time, {:.5f} sec.".format(clock_t1 - clock_t0, cpu_t1 - cpu_t0))
    res_lst = np.concatenate(res_lst, axis=0)
    np.savetxt(out_file, res_lst, header='snp_0 snp_1 eff var chi', comments='')
    return res_lst


def remma_epiAD_pair_cpu(y, xmat, zmat, gmat_lst, var_com, bed_file, snp_pair_file, max_test_pair=50000, p_cut=1.0e-4, out_file='remma_epiAD_pair_cpu'):
    """
    加加上位检验
    :param y: 表型
    :param xmat: 固定效应设计矩阵
    :param zmat: 随机效应设计矩阵，csr稀疏矩阵
    :param gmat_lst: 基因组关系矩阵列表
    :param var_com: 方差组分
    :param bed_file: plink文件
    :param snp_pair_file: SNP互作对文件，取前两列，最小值为0，最大值为num_snp-1
    :param max_test_pair: 分批次检验，每次检验的最大互作对数
    :param p_cut: 阈值
    :param out_file: 输出文件
    :return:
    """
    logging.info("计算V矩阵及其逆矩阵")
    y = np.array(y).reshape(-1, 1)
    n = y.shape[0]
    xmat = np.array(xmat).reshape(n, -1)
    vmat = np.diag([var_com[-1]] * n)
    for val in range(len(gmat_lst)):
        vmat += zmat.dot((zmat.dot(gmat_lst[val])).T) * var_com[val]
    del gmat_lst
    gc.collect()
    vmat_inv = np.linalg.inv(vmat)
    logging.info("计算P矩阵")
    vxmat = np.dot(vmat_inv, xmat)
    xvxmat = np.dot(xmat.T, vxmat)
    xvxmat = np.linalg.inv(xvxmat)
    pmat = reduce(np.dot, [vxmat, xvxmat, vxmat.T])
    pmat = vmat_inv - pmat
    pymat = zmat.T.dot(np.dot(pmat, y))
    pvpmat = reduce(np.dot, [pmat, vmat, pmat])
    pvpmat = zmat.T.dot((zmat.T.dot(pvpmat)).T)
    del vmat, vmat_inv, pmat
    gc.collect()
    logging.info("读取SNP文件")
    snp_mat = read_plink(bed_file)
    num_id, num_snp = snp_mat.shape
    if np.any(np.isnan(snp_mat)):
        logging.warning('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    freq = np.sum(snp_mat, axis=0) / (2 * num_id)
    freq.shape = (1, -1)
    snp_matA = snp_mat - 2 * freq
    snp_mat[snp_mat > 1.5] = 0.0  # 2替换为0, 变为0、1、0编码
    snp_matD = snp_mat - 2 * freq * (1 - freq)
    del snp_mat
    gc.collect()
    logging.info("开始检验")
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
        epi_mat = snp_matA[:, snp_pair[:, 0]] * snp_matD[:, snp_pair[:, 1]]
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


def remma_epiAD_eff_cpu(y, xmat, zmat, gmat_lst, var_com, bed_file, snp_lst_0=None, eff_cut=-999.0, out_file='remma_epiAD_eff_cpu'):
    """
    加加上位检验
    :param y: 表型
    :param xmat: 固定效应设计矩阵
    :param zmat: 随机效应设计矩阵，csr稀疏矩阵
    :param gmat_lst: 基因组关系矩阵列表
    :param var_com: 方差组分
    :param bed_file: plink文件
    :param snp_lst_0: 互作对第一个SNP列表，最小值为0，最大值为num_snp-1
    :param eff_cut: 依据阈值保留的互作对
    :param out_file: 输出文件
    :return:
    """
    logging.info("计算V矩阵及其逆矩阵")
    y = np.array(y).reshape(-1, 1)
    n = y.shape[0]
    xmat = np.array(xmat).reshape(n, -1)
    vmat = np.diag([var_com[-1]] * n)
    for val in range(len(gmat_lst)):
        vmat += gmat_lst[val] * var_com[val]
    del gmat_lst
    gc.collect()
    vmat_inv = np.linalg.inv(vmat)
    logging.info("计算P矩阵")
    vxmat = np.dot(vmat_inv, xmat)
    xvxmat = np.dot(xmat.T, vxmat)
    xvxmat = np.linalg.inv(xvxmat)
    pmat = reduce(np.dot, [vxmat, xvxmat, vxmat.T])
    pmat = vmat_inv - pmat
    pymat = zmat.T.dot(np.dot(pmat, y))
    del vmat, vmat_inv, pmat
    gc.collect()
    logging.info("读取SNP文件")
    snp_mat = read_plink(bed_file)
    num_snp = snp_mat.shape[1]
    num_id = snp_mat.shape[0]
    if np.any(np.isnan(snp_mat)):
        logging.warning('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    freq = np.sum(snp_mat, axis=0) / (2 * num_id)
    freq.shape = (1, -1)
    snp_matA = snp_mat - 2 * freq
    snp_mat[snp_mat > 1.5] = 0.0  # 2替换为0, 变为0、1、0编码
    snp_matD = snp_mat - 2 * freq * (1 - freq)
    del snp_mat
    gc.collect()
    logging.info('检验')
    if snp_lst_0 is None:
        snp_lst_0 = range(num_snp)
    else:
        if max(snp_lst_0) >= num_snp or min(snp_lst_0) < 0:
            logging.error('snp_lst_0 is out of range!')
            sys.exit()
    snp_lst_1 = list(range(num_snp))
    clock_t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    res_lst = []
    for i in snp_lst_0:
        snp_lst_11 = snp_lst_1[:]
        snp_lst_11.remove(i)
        epi_mat = snp_matA[:, i:(i+1)] * snp_matD[:, snp_lst_11]
        eff_vec = np.dot(epi_mat.T, pymat)
        res = np.concatenate([np.array([i]*len(snp_lst_11)).reshape(-1, 1), np.array(snp_lst_11).reshape(-1, 1), eff_vec], axis=1)
        res_lst.append(res[np.abs(res[:, -1]) > eff_cut, :])
    clock_t1 = time.perf_counter()
    cpu_t1 = time.process_time()
    logging.info("Running time: Clock time, {:.5f} sec; CPU time, {:.5f} sec.".format(clock_t1 - clock_t0, cpu_t1 - cpu_t0))
    res_lst = np.concatenate(res_lst, axis=0)
    np.savetxt(out_file, res_lst, header='snp_0 snp_1 eff', comments='')
    return res_lst


from _cremma_epi_eff_cpu import ffi, lib


def remma_epiAD_eff_cpu_c(y, xmat, zmat, gmat_lst, var_com, bed_file, snp_lst_0=None, eff_cut=-999.0, out_file='remma_epiAD_eff_cpu_c'):
    """
    加加上位检验
    :param y: 表型
    :param xmat: 固定效应设计矩阵
    :param zmat: 随机效应设计矩阵，csr稀疏矩阵
    :param gmat_lst: 基因组关系矩阵列表
    :param var_com: 方差组分
    :param bed_file: plink文件
    :param snp_lst_0: 互作对第一个SNP列表，最小值为0，最大值为num_snp-1
    :param eff_cut: 依据阈值保留的互作对
    :param out_file: 输出文件
    :return:
    """
    logging.info("计算V矩阵及其逆矩阵")
    y = np.array(y).reshape(-1, 1)
    n = y.shape[0]
    xmat = np.array(xmat).reshape(n, -1)
    vmat = np.diag([var_com[-1]] * n)
    for val in range(len(gmat_lst)):
        vmat += zmat.dot((zmat.dot(gmat_lst[val])).T) * var_com[val]
    del gmat_lst
    gc.collect()
    vmat_inv = np.linalg.inv(vmat)
    logging.info("计算P矩阵")
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
        snp_lst_0 = range(num_snp)
    else:
        if max(snp_lst_0) > num_snp - 1 or min(snp_lst_0) < 0:
            logging.error('snp_lst_0 is out of range!')
            sys.exit()
    logging.info("python变量转化到C")
    pbed_file = ffi.new("char[]", bed_file.encode('ascii'))
    pnum_id = ffi.cast("long long", num_id)
    pnum_snp = ffi.cast("long long", num_snp)
    snp_lst_0 = np.array(list(snp_lst_0), dtype=np.longlong)
    psnp_lst_0 = ffi.cast("long long *", snp_lst_0.ctypes.data)
    # psnp_lst_0 = ffi.cast("long long *", ffi.from_buffer(snp_lst_0))
    plen_snp_lst_0 = ffi.cast("long long", len(snp_lst_0))
    ppymat = ffi.cast("double *", pymat.ctypes.data)
    peff_cut = ffi.cast("double", eff_cut)
    pout_file = ffi.new("char[]", out_file.encode('ascii'))
    logging.info('检验')
    clock_t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    lib.remma_epiAD_eff_cpu(pbed_file, pnum_id, pnum_snp, psnp_lst_0, plen_snp_lst_0, ppymat, peff_cut, pout_file)
    clock_t1 = time.perf_counter()
    cpu_t1 = time.process_time()
    logging.info("Running time: Clock time, {:.5f} sec; CPU time, {:.5f} sec.".format(clock_t1 - clock_t0, cpu_t1 - cpu_t0))
    return 0

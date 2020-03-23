import numpy as np
import pandas as pd
from functools import reduce
import gc
import time
import logging
from scipy.stats import chi2

from gmat.process_plink.process_plink import read_plink, impute_geno


def remma_epiAA_eff_gpu(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, max_test_pair=50000, eff_cut=-999.0, out_file='remma_epiAA_eff_gpu'):
    """
    加加上位检验，GPU加速
    :param y: 表型
    :param xmat: 固定效应设计矩阵
    :param gmat_lst: 基因组关系矩阵列表
    :param var_com: 方差组分
    :param bed_file: plink文件
    :param snp_lst_0: 互作对第一个SNP列表
    :param max_test_pair: 最大检验互作对数
    :param eff_cut: 依据阈值保留的互作对
    :param out_file: 输出文件
    :return:
    """
    try:
        import cupy as cp
    except Exception as e:
        logging.error(e)
        return e
    logging.info("计算V矩阵及其逆矩阵")
    y = np.array(y).reshape(-1, 1)
    n = y.shape[0]
    xmat = np.array(xmat).reshape(n, -1)
    vmat = np.diag([var_com[-1]] * n)
    for val in range(len(gmat_lst)):
        vmat += gmat_lst[val] * var_com[val]
    vmat_inv = np.linalg.inv(vmat)
    logging.info("计算P矩阵")
    vxmat = np.dot(vmat_inv, xmat)
    xvxmat = np.dot(xmat.T, vxmat)
    xvxmat = np.linalg.inv(xvxmat)
    pmat = reduce(np.dot, [vxmat, xvxmat, vxmat.T])
    pmat = vmat_inv - pmat
    pymat = np.dot(pmat, y)
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
    snp_mat = snp_mat - 2 * freq
    logging.info('检验')
    if snp_lst_0 is None:
        snp_lst_0 = range(num_snp - 1)
    else:
        if max(snp_lst_0) >= num_snp - 1 or min(snp_lst_0) < 0:
            logging.error('snp_lst_0 is out of range!')
            sys.exit()
    snp_mat0 = cp.array(snp_mat[:, snp_lst_0])
    pymat = cp.array(pymat)
    clock_t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    res_lst = []
    start, end = 0, 0
    while True:
        start = end
        if start >= num_snp:
            break
        end = start + max_test_pair
        if end >= num_snp:
            end = num_snp
        snp_mat1 = cp.array(snp_mat[:, start:end])
        num_snp1 = snp_mat1.shape[1]
        for i in range(len(snp_lst_0)):
            if end >= i+2 and start <= i:
                epi_mat = snp_mat0[:, i:(i + 1)] * snp_mat1[:, (i+1):]
                eff_vec = cp.dot(epi_mat.T, pymat)
                res = cp.concatenate([cp.array([snp_lst_0[i]] * (snp_mat1.shape[1] - i - 1)).reshape(-1, 1),
                                      cp.arange(i+1, snp_mat1.shape[1]).reshape(-1, 1), eff_vec], axis=1)
                res_lst.append(res[cp.absolute(res[:, -1]) > eff_cut, :])
            elif start > i:
                epi_mat = snp_mat0[:, i:(i + 1)] * snp_mat1
                eff_vec = cp.dot(epi_mat.T, pymat)
                res = cp.concatenate([cp.array([snp_lst_0[i]] * num_snp1).reshape(-1, 1),
                                  cp.arange(start, end).reshape(-1, 1), eff_vec], axis=1)
                res_lst.append(res[cp.absolute(res[:, -1]) > eff_cut, :])
            else:
                continue
    clock_t1 = time.perf_counter()
    cpu_t1 = time.process_time()
    logging.info("Running time: Clock time, {:.5f} sec; CPU time, {:.5f} sec.".format(clock_t1 - clock_t0, cpu_t1 - cpu_t0))
    res_lst = cp.asnumpy(cp.concatenate(res_lst, axis=0))
    np.savetxt(out_file, res_lst, header='snp_0 snp_1 eff', comments='')
    return res_lst

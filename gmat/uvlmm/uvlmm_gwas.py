import numpy as np
import pandas as pd
from functools import reduce
import gc
import time
from scipy.stats import chi2

from gmat.process_plink.process_plink import read_plink, impute_geno


def uvlmm_gwas_add(y, xmat, gmat_lst, var_com, bed_file):
    """
    将SNP效应当成固定效应的单标记检验，可以包含多个关系矩阵（必含加性关系矩阵）
    :param y:
    :param xmat:
    :param gmat_lst:
    :param var_com:
    :param bed_file:
    :return:
    """
    # 计算V矩阵、逆矩阵
    y = np.array(y).reshape(-1, 1)
    n = y.shape[0]
    xmat = np.array(xmat).reshape(n, -1)
    vmat = np.diag([var_com[-1]] * n)
    for val in range(len(gmat_lst)):
        vmat += gmat_lst[val] * var_com[val]
    vmat = np.linalg.inv(vmat)
    # 读取SNP
    snp_mat = read_plink(bed_file)
    if np.any(np.isnan(snp_mat)):
        print('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    freq = np.sum(snp_mat, axis=0) / (2 * snp_mat.shape[0])
    freq.shape = (1, snp_mat.shape[1])
    snp_mat = snp_mat - 2 * freq
    # 检验
    eff_vec = []
    chi_vec = []
    p_vec = []
    for i in range(snp_mat.shape[1]):
        xmat_snp = np.concatenate([xmat, snp_mat[:, i:(i+1)]], axis=1)
        vxmat = np.dot(vmat, xmat_snp)
        xvxmat = np.dot(xmat_snp.T, vxmat)
        xvxmat = np.linalg.inv(xvxmat)
        snp_eff = np.dot(xvxmat, np.dot(vxmat.T, y))[-1, -1]
        snp_var = xvxmat[-1, -1]
        snp_chi = snp_eff*snp_eff/snp_var
        p_val = chi2.sf(snp_chi, 1)
        eff_vec.append(snp_eff)
        chi_vec.append(snp_chi)
        p_vec.append(p_val)
    snp_info_file = bed_file + '.bim'
    snp_info = pd.read_csv(snp_info_file, sep='\s+', header=None)
    res_df = snp_info.iloc[:, [0, 1, 3, 4, 5]]
    res_df.columns = ['chro', 'snp_ID', 'pos', 'allele1', 'allele2']
    res_df.loc[:, 'eff_val'] = eff_vec
    res_df.loc[:, 'chi_val'] = chi_vec
    res_df.loc[:, 'p_val'] = p_vec
    return res_df


def uvlmm_gwas_add_eigen(y, xmat, agmat, var_com, bed_file):
    """
    将SNP效应当成固定效应的单标记检验，仅包含加性基因组关系矩阵，特征分解加快运算速度
    :param y:
    :param xmat:
    :param agmat:
    :param var_com:
    :param bed_file:
    :return:
    """


def uvlmm_gwas_dom(y, xmat, gmat_lst, var_com, bed_file):
    """
    显性检验，可以包含多个关系矩阵（必含加性关系矩阵）
    :param y:
    :param xmat:
    :param gmat_lst:
    :param var_com:
    :param bed_file:
    :return:
     """
    # 计算V矩阵、逆矩阵
    y = np.array(y).reshape(-1, 1)
    n = y.shape[0]
    xmat = np.array(xmat).reshape(n, -1)
    vmat = np.diag([var_com[-1]] * n)
    for val in range(len(gmat_lst)):
        vmat += gmat_lst[val] * var_com[val]
    vmat = np.linalg.inv(vmat)
    # 读取SNP
    snp_mat = read_plink(bed_file)
    if np.any(np.isnan(snp_mat)):
        print('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    freq = np.sum(snp_mat, axis=0) / (2 * snp_mat.shape[0])
    freq.shape = (1, snp_mat.shape[1])
    snp_mat_add = snp_mat - 2 * freq
    snp_mat[snp_mat > 1.5] = 0.0
    snp_mat = snp_mat - 2 * freq * (1 - freq)
    # 检验
    eff_vec = []
    chi_vec = []
    p_vec = []
    for i in range(snp_mat.shape[1]):
        xmat_snp = np.concatenate([xmat, snp_mat_add[:, i:(i+1)], snp_mat[:, i:(i+1)]], axis=1)
        vxmat = np.dot(vmat, xmat_snp)
        xvxmat = np.dot(xmat_snp.T, vxmat)
        xvxmat = np.linalg.inv(xvxmat)
        snp_eff = np.dot(xvxmat, np.dot(vxmat.T, y))[-1, -1]
        snp_var = xvxmat[-1, -1]
        snp_chi = snp_eff*snp_eff/snp_var
        p_val = chi2.sf(snp_chi, 1)
        eff_vec.append(snp_eff)
        chi_vec.append(snp_chi)
        p_vec.append(p_val)
    snp_info_file = bed_file + '.bim'
    snp_info = pd.read_csv(snp_info_file, sep='\s+', header=None)
    res_df = snp_info.iloc[:, [0, 1, 3, 4, 5]]
    res_df.columns = ['chro', 'snp_ID', 'pos', 'allele1', 'allele2']
    res_df.loc[:, 'eff_val'] = eff_vec
    res_df.loc[:, 'chi_val'] = chi_vec
    res_df.loc[:, 'p_val'] = p_vec
    return res_df

def uvlmm_gwas_dom_eigen(y, xmat, agmat, var_com, bed_file):
    """
    显性检验，仅包含加性基因组关系矩阵，特征分解加快运算速度
    :param y:
    :param xmat:
    :param agmat:
    :param var_com:
    :param bed_file:
    :return:
    """


def uvlmm_gwas_epiAA(y, xmat, gmat_lst, var_com, bed_file):
    """
    加加上位，将SNP效应当成固定效应的单标记检验，可以包含多个关系矩阵（必含加性关系矩阵）
    :param y:
    :param xmat:
    :param gmat_lst:
    :param var_com:
    :param bed_file:
    :return:
    """
    # 计算V矩阵、逆矩阵
    y = np.array(y).reshape(-1, 1)
    n = y.shape[0]
    xmat = np.array(xmat).reshape(n, -1)
    vmat = np.diag([var_com[-1]] * n)
    for val in range(len(gmat_lst)):
        vmat += gmat_lst[val] * var_com[val]
    vmat = np.linalg.inv(vmat)
    # 读取SNP
    snp_mat = read_plink(bed_file)
    if np.any(np.isnan(snp_mat)):
        print('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    freq = np.sum(snp_mat, axis=0) / (2 * snp_mat.shape[0])
    freq.shape = (1, snp_mat.shape[1])
    snp_mat = snp_mat - 2 * freq
    # 检验
    snpi = []
    snpj = []
    eff_vec = []
    chi_vec = []
    p_vec = []
    for i in range(snp_mat.shape[1]-1):
        for j in range(i+1, snp_mat.shape[1]):
            snpi.append(i)
            snpj.append(j)
            xmat_snp = np.concatenate([xmat, snp_mat[:, i:(i + 1)], snp_mat[:, j:(j + 1)], snp_mat[:, i:(i + 1)]*snp_mat[:, j:(j + 1)]], axis=1)
            vxmat = np.dot(vmat, xmat_snp)
            xvxmat = np.dot(xmat_snp.T, vxmat)
            xvxmat = np.linalg.inv(xvxmat)
            snp_eff = np.dot(xvxmat, np.dot(vxmat.T, y))[-1, -1]
            snp_var = xvxmat[-1, -1]
            snp_chi = snp_eff * snp_eff / snp_var
            p_val = chi2.sf(snp_chi, 1)
            eff_vec.append(snp_eff)
            chi_vec.append(snp_chi)
            p_vec.append(p_val)
    res_df = pd.DataFrame({
        'snpi': snpi,
        'snpj': snpj,
        'snp_eff': eff_vec,
        'p_val': p_vec
    })
    return res_df

import numpy as np
import pandas as pd
from scipy import linalg
import logging
from tqdm import tqdm
from functools import reduce
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.process_plink.process_plink import read_plink, impute_geno


def lm_snp_eff(pheno_file, bed_file, out_file='lm_snp_eff'):
    """
    :param pheno_file: phenotypic file. The fist two columns are family id, individual id which are same as plink *.fam
    file. The third column is always ones for population mean. The last column is phenotypic values. The ohter covariate
    can be added between columns for population mean and phenotypic values.
    :param bed_file: the prefix for binary file
    :param out_file: the output file
    :return: the estimated snp effect
    """
    y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)
    snp_mat = read_plink(bed_file)
    if np.any(np.isnan(snp_mat)):
        logging.warning('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    snp_eff = []
    for i in tqdm(range(snp_mat.shape[1])):
        xmati = np.concatenate([xmat, snp_mat[:, i:(i+1)]], axis=1)
        eff = np.dot(linalg.inv(np.dot(xmati.T, xmati)), np.dot(xmati.T, y))
        snp_eff.append(eff[-1, -1])
    df = pd.read_csv(bed_file + '.bim', sep='\s+', header=None)
    df['eff'] = snp_eff
    df.to_csv(out_file, sep=' ', header=False, index=False)


def lm_pred(pheno_file, bed_file, agmat, out_file='lm_pred'):
    """
    :param pheno_file: phenotypic file. The fist two columns are family id, individual id which are same as plink *.fam
    file. The third column is always ones for population mean. The last column is phenotypic values. The ohter covariate
    can be added between columns for population mean and phenotypic values.
    :param bed_file: the prefix for binary file
    :param agmat: additive genomic relationship matrix
    :param out_file: the output file
    :return: the estimated snp effect
    """
    y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)
    vmat = np.diag([1] * y.shape[0])
    vxmat = np.dot(vmat, xmat)
    xvxmat = np.dot(xmat.T, vxmat)
    xvxmat = np.linalg.inv(xvxmat)
    pmat = reduce(np.dot, [vxmat, xvxmat, vxmat.T])
    pmat = vmat - pmat
    zpymat = zmat.T.dot(np.dot(pmat, y))
    eff = np.dot(agmat, zpymat)
    np.savetxt(out_file + '.rand_eff', eff)

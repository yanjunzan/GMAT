import numpy as np
from pandas_plink import read_plink1_bin
from tqdm import tqdm
import logging


def read_plink(bed_file):
    snp_info = read_plink1_bin(bed_file + ".bed", bed_file + ".bim", bed_file + ".fam", verbose=False)
    return snp_info.values


def impute_geno(snp_mat):
    ind_na = np.where(np.isnan(snp_mat))
    col_na = set(ind_na[1])  # get col number where na exist
    for i in tqdm(col_na):
        snpi = snp_mat[:, i]
        code0 = np.sum(np.absolute(snpi - 0.0) < 1e-10)
        code1 = np.sum(np.absolute(snpi - 1.0) < 1e-10)
        code2 = np.sum(np.absolute(snpi - 2.0) < 1e-10)
        code_count = code0 + code1 + code2
        p_lst = [code0/code_count, code1/code_count, code2/code_count]
        icol_na = np.where(np.isnan(snpi))
        snpi[icol_na] = np.random.choice([0.0, 1.0, 2.0], len(icol_na[0]), p=p_lst)
        snp_mat[:, i] = snpi
    return snp_mat


def shuffle_bed(bed_file):
    """
    shuffle the genotypes of individuals snp-by-snp
    :param bed_file: the prefix for plink binary file
    :return: the shuffled plink binary file
    """
    try:
        from pysnptools.snpreader import Bed
    except Exception as e:
        print(e)
        return 0
    logging.INFO('Read the plink file')
    data = Bed(bed_file, count_A1=False).read()
    num_snp = data.val.shape[1]
    logging.INFO("Start shuffle the genotypes snp-by-snp")
    for i in tqdm(range(num_snp)):
        np.random.shuffle(data.val[:, i])
    logging.INFO('Write the shuffled plink file')
    Bed.write(bed_file + '_shuffle', data, count_A1=False)
    return 1

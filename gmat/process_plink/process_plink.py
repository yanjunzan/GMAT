import numpy as np
from pandas_plink import read_plink1_bin



def read_plink(bed_file):
    snp_info = read_plink1_bin(bed_file + ".bed", bed_file + ".bim", bed_file + ".fam", verbose=False)
    return snp_info.values


def impute_geno(snp_mat):
    ind = np.where(np.isnan(snp_mat))
    ind_len = len(ind[0])
    snp_mat.val[ind] = np.random.choice([0.0, 1.0, 2.0], ind_len)
    return snp_mat


def shuffle_bed(bed_file):
    try:
        from pysnptools.snpreader import Bed
    except Exception as e:
        print(e)
        return 0
    data = Bed(bed_file, count_A1=False).read()
    num_snp = data.val.shape[1]
    for i in range(num_snp):
        np.random.shuffle(data.val[:, i])
    Bed.write(bed_file + '_shuffle', data, count_A1=False)
    return 1

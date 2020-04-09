import os
import logging
import pandas as pd
import numpy as np
from gmat.remma.random_pair import random_pair
from gmat.remma.remma_epiAA.remma_epiAA_pair import remma_epiAA_pair
from gmat.remma.remma_epiAA.remma_epiAA_eff import remma_epiAA_eff, remma_epiAA_eff_parallel


def remma_epiAA_approx(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, num_random_pair=100000, out_file='epiAA_approx'):
    """
    additive by additive epistasis test by random SNP-BLUP model based on approximate test
    :param pheno_file: phenotypic file. The fist two columns are family id, individual id which are same as plink *.fam
    file. The third column is always ones for population mean. The last column is phenotypic values. The ohter covariate
    can be added between columns for population mean and phenotypic values.
    :param bed_file: the prefix for binary file
    :param gmat_lst: a list of genomic relationship matrixes.
    :param var_com: Estimated variances
    :param p_cut: put cut value. default value is 1.0e-5.
    :param num_random_pair: the number of random pairs to estimate the approximate variances of estimated effects. Default value is 100,000
    :param out_file: The prefix for output file. Default value is 'epiAA_approx'
    :return: 0
    """
    logging.info("\n\n#####Randomly select {:d} pairs, and test these SNP pairs#####".format(num_random_pair))
    snp_df = pd.read_csv(bed_file + '.bim', header=None, sep='\s+')
    num_snp = snp_df.shape[0]  # the number of snp
    random_pair(num_snp, out_file=out_file + '.random_pair', num_pair=num_random_pair)
    remma_epiAA_pair(pheno_file, bed_file, gmat_lst, var_com, snp_pair_file=out_file + '.random_pair', p_cut=1,
                     out_file=out_file + '.random')
    res_df = pd.read_csv(out_file + '.random', header=0, sep='\s+')
    var_median = np.median(res_df['var'])
    os.remove(out_file + '.random_pair')
    os.remove(out_file + '.random')
    logging.info("\n\n#####Screen the epistatic effects and select top SNP pairs based on approximate test#####")
    remma_epiAA_eff(pheno_file, bed_file, gmat_lst, var_com, var_app=var_median, p_cut=p_cut,
                    out_file=out_file + '.approx_p')
    logging.info("\n\n#####Calculate exact p values for top SNP pairs#####")
    remma_epiAA_pair(pheno_file, bed_file, gmat_lst, var_com, snp_pair_file=out_file + '.approx_p', p_cut=1,
                     out_file=out_file + '.exact_p')
    logging.info("\n\n#####Merge the results#####")
    p_dct = {}
    with open(out_file + '.approx_p', 'r') as fin:
        for line in fin:
            arr = line.split()
            p_dct[' '.join(arr[:2])] = arr[-1]
    with open(out_file + '.exact_p', 'r') as fin, open(out_file, 'w') as fout:
        for line in fin:
            arr = line.split()
            arr.insert(-1, p_dct[' '.join(arr[:2])])
            fout.write(' '.join(arr) + '\n')
    os.remove(out_file + '.approx_p')
    os.remove(out_file + '.exact_p')
    return 0


def remma_epiAA_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel, p_cut=1.0e-5, num_random_pair=100000, out_file='epiAA_approx'):
    """
    additive by additive epistasis test by random SNP-BLUP model based on approximate test
    :param pheno_file: phenotypic file. The fist two columns are family id, individual id which are same as plink *.fam
    file. The third column is always ones for population mean. The last column is phenotypic values. The ohter covariate
    can be added between columns for population mean and phenotypic values.
    :param bed_file: the prefix for binary file
    :param gmat_lst: a list of genomic relationship matrixes.
    :param var_com: Estimated variances
    :param parallel: A list containing two integers. The first integer is the number of parts to parallel. The second
    integer is the part to run. For example, parallel = [3, 1], parallel = [3, 2] and parallel = [3, 3] mean to divide
    :param p_cut: put cut value. default value is 1.0e-5.
    :param num_random_pair: the number of random pairs to estimate the approximate variances of estimated effects. Default value is 100,000
    :param out_file: The prefix for output file. Default value is 'epiAA_approx'
    :return: 0
    """
    logging.info("\n\n#####Randomly select {:d} pairs, and test these SNP pairs#####".format(num_random_pair))
    bim_df = pd.read_csv(bed_file + '.bim', header=None)
    num_snp = bim_df.shape[0]
    random_pair(num_snp, out_file=out_file + '.random_pair.' + str(parallel[1]), num_pair=num_random_pair)
    remma_epiAA_pair(pheno_file, bed_file, gmat_lst, var_com, snp_pair_file=out_file + '.random_pair.' + str(parallel[1]), p_cut=1,
                     out_file=out_file + '.random.' + str(parallel[1]))
    res_df = pd.read_csv(out_file + '.random.' + str(parallel[1]), header=0, sep='\s+')
    var_median = np.median(res_df['var'])
    os.remove(out_file + '.random_pair.' + str(parallel[1]))
    os.remove(out_file + '.random.' + str(parallel[1]))
    logging.info("\n\n#####Screen the epistatic effects and select top SNP pairs based on approximate test#####")
    remma_epiAA_eff_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=parallel, var_app=var_median, p_cut=p_cut,
                    out_file=out_file + '.approx_p')
    logging.info("\n\n#####Calculate exact p values for top SNP pairs#####")
    remma_epiAA_pair(pheno_file, bed_file, gmat_lst, var_com, snp_pair_file=out_file + '.approx_p.' + str(parallel[1]), p_cut=1,
                     out_file=out_file + '.exact_p.' + str(parallel[1]))
    logging.info("\n\n#####Merge the results#####")
    p_dct = {}
    with open(out_file + '.approx_p.' + str(parallel[1]), 'r') as fin:
        for line in fin:
            arr = line.split()
            p_dct[' '.join(arr[:2])] = arr[-1]
    with open(out_file + '.exact_p.' + str(parallel[1]), 'r') as fin, open(out_file + '.' + str(parallel[1]), 'w') as fout:
        for line in fin:
            arr = line.split()
            arr.insert(-1, p_dct[' '.join(arr[:2])])
            fout.write(' '.join(arr) + '\n')
    os.remove(out_file + '.approx_p.' + str(parallel[1]))
    os.remove(out_file + '.exact_p.' + str(parallel[1]))
    return 0

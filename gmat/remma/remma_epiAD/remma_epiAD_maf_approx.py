import os
import logging
import pandas as pd
import numpy as np
import gc
from gmat.remma.random_pair import random_pairAD
from gmat.remma.remma_epiAD.remma_epiAD_pair import remma_epiAD_pair
from gmat.remma.remma_epiAD.remma_epiAD_maf_eff import remma_epiAD_maf_eff, remma_epiAD_maf_eff_parallel
from gmat.process_plink.process_plink import read_plink, impute_geno


def remma_epiAD_maf_approx(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, num_random_pair=100000, out_file='epiAD_maf_approx'):
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
    :param out_file: The prefix for output file. Default value is 'epiAA_maf_approx'
    :return: 0
    """
    logging.info("\n\n#####Randomly select {:d} pairs, and test these SNP pairs#####".format(num_random_pair))
    snp_df = pd.read_csv(bed_file + '.bim', header=None, sep='\s+')
    num_snp = snp_df.shape[0]  # the number of snp
    random_pairAD(num_snp, out_file=out_file + '.random_pairAD', num_pair=num_random_pair)
    remma_epiAD_pair(pheno_file, bed_file, gmat_lst, var_com, snp_pair_file=out_file + '.random_pairAD', p_cut=1,
                     out_file=out_file + '.random')
    os.remove(out_file + '.random_pairAD')
    logging.info("\n\n#####Calcualte the approximate denominator for Wald chi-square test#####")
    snp_mat = read_plink(bed_file)
    num_id, num_snp = snp_mat.shape
    if np.any(np.isnan(snp_mat)):
        logging.warning('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    freqA = np.sum(snp_mat, axis=0) / (2 * num_id)
    freqA[freqA > 0.5] = 1 - freqA[freqA > 0.5]  # maf
    np.savetxt(out_file + '.maf', freqA)
    freqA = np.array(freqA*20, dtype=np.longlong)
    freqA_dct = dict(zip(np.array(list(range(len(freqA))), dtype=str), freqA))
    freqD = np.sum(np.absolute(snp_mat-1.0) < 0.001, axis=0) / num_id
    del snp_mat
    gc.collect()
    freqD[freqD > 0.5] = 1 - freqD[freqD > 0.5]
    np.savetxt(out_file + '.heter', freqD)
    freqD = np.array(freqD * 20, dtype=np.longlong)
    freqD_dct = dict(zip(np.array(list(range(len(freqD))), dtype=str), freqD))
    median_dct = {}
    median_count_dct = {}
    with open(out_file + '.random') as fin:
        fin.readline()
        for line in fin:
            arr = line.split()
            key_val = ' '.join([str(freqA_dct[arr[0]]), str(freqD_dct[arr[1]])])
            median_count_dct[key_val] = median_count_dct.get(key_val, 0) + 1
            median_dct[key_val] = median_dct.get(key_val, 0.0) + float(arr[-3])
    all_median = 0
    all_count = 0
    for val in median_count_dct:
        all_median += median_dct[val]
        all_count += median_count_dct[val]
        median_dct[val] = median_dct[val] / median_count_dct[val]
    all_median = all_median / all_count
    freq_deno = np.ones(111)
    with open(out_file + '.freq_denominator', 'w') as fout:
        for key1 in set(freqA):
            for key2 in set(freqD):
                key_val = ' '.join([str(key1), str(key2)])
                if key_val not in median_dct:
                    median_dct[key_val] = all_median
                fout.write(key_val + ' ' + str(median_dct[key_val]) + '\n')
                freq_deno[key1 * 10 + key2] = median_dct[key_val]
    logging.info("\n\n#####Screen the epistatic effects and select top SNP pairs based on approximate test#####")
    remma_epiAD_maf_eff(pheno_file, bed_file, gmat_lst, var_com, freqA=freqA, freqD=freqD, freq_deno=freq_deno, p_cut=p_cut,
                    out_file=out_file + '.approx_p')
    logging.info("\n\n#####Calculate exact p values for top SNP pairs#####")
    remma_epiAD_pair(pheno_file, bed_file, gmat_lst, var_com, snp_pair_file=out_file + '.approx_p', p_cut=1,
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



def remma_epiAD_maf_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel, p_cut=1.0e-5, num_random_pair=100000, out_file='epiAD_maf_approx_parallel'):
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
    random_pairAD(num_snp, out_file=out_file + '.random_pair.' + str(parallel[1]), num_pair=num_random_pair)
    remma_epiAD_pair(pheno_file, bed_file, gmat_lst, var_com, snp_pair_file=out_file + '.random_pair.' + str(parallel[1]), p_cut=1,
                     out_file=out_file + '.random.' + str(parallel[1]))
    os.remove(out_file + '.random_pair.' + str(parallel[1]))
    logging.info("\n\n#####Calcualte the approximate denominator for Wald chi-square test#####")
    snp_mat = read_plink(bed_file)
    num_id, num_snp = snp_mat.shape
    if np.any(np.isnan(snp_mat)):
        logging.warning('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    freqA = np.sum(snp_mat, axis=0) / (2 * num_id)
    freqA[freqA > 0.5] = 1 - freqA[freqA > 0.5]  # maf
    np.savetxt(out_file + '.maf.' + str(parallel[1]), freqA)
    freqA = np.array(freqA * 20, dtype=np.longlong)
    freqA_dct = dict(zip(np.array(list(range(len(freqA))), dtype=str), freqA))
    freqD = np.sum(np.absolute(snp_mat - 1.0) < 0.001, axis=0) / num_id
    del snp_mat
    gc.collect()
    freqD[freqD > 0.5] = 1 - freqD[freqD > 0.5]
    np.savetxt(out_file + '.heter.' + str(parallel[1]), freqD)
    freqD = np.array(freqD * 20, dtype=np.longlong)
    freqD_dct = dict(zip(np.array(list(range(len(freqD))), dtype=str), freqD))
    median_dct = {}
    median_count_dct = {}
    with open(out_file + '.random.' + str(parallel[1])) as fin:
        fin.readline()
        for line in fin:
            arr = line.split()
            key_val = ' '.join([str(freqA_dct[arr[0]]), str(freqD_dct[arr[1]])])
            median_count_dct[key_val] = median_count_dct.get(key_val, 0) + 1
            median_dct[key_val] = median_dct.get(key_val, 0.0) + float(arr[-3])
    all_median = 0
    all_count = 0
    for val in median_count_dct:
        all_median += median_dct[val]
        all_count += median_count_dct[val]
        median_dct[val] = median_dct[val] / median_count_dct[val]
    all_median = all_median / all_count
    freq_deno = np.ones(111)
    with open(out_file + '.freq_denominator.' + str(parallel[1]), 'w') as fout:
        for key1 in set(freqA):
            for key2 in set(freqD):
                key_val = ' '.join([str(key1), str(key2)])
                if key_val not in median_dct:
                    median_dct[key_val] = all_median
                fout.write(key_val + ' ' + str(median_dct[key_val]) + '\n')
                freq_deno[key1 * 10 + key2] = median_dct[key_val]
    logging.info("\n\n#####Screen the epistatic effects and select top SNP pairs based on approximate test#####")
    remma_epiAD_maf_eff_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=parallel,
                                 freqA=freqA, freqD=freqD, freq_deno=freq_deno, p_cut=p_cut, out_file=out_file + '.approx_p')
    logging.info("\n\n#####Calculate exact p values for top SNP pairs#####")
    remma_epiAD_pair(pheno_file, bed_file, gmat_lst, var_com, snp_pair_file=out_file + '.approx_p.' + str(parallel[1]), p_cut=1,
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

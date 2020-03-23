import sys
import logging
import numpy as np
from scipy import linalg
import pandas as pd
import time
from gmat.process_plink.process_plink import read_plink, impute_geno


def output_mat(mat, id, out_file, out_fmt):
    if out_fmt is 'mat':
        np.savez(out_file + '0', mat=mat)
    elif out_fmt is 'row_col_val':
        ind = np.tril_indices_from(mat)
        np.savez(out_file + '1', row=ind[0], col=ind[1], val=mat[ind])
    elif out_fmt is 'id_id_val':
        ind = np.tril_indices_from(mat)
        np.savez(out_file + '2', id0=id[ind[0]], id1=id[ind[1]], val=mat[ind])
    else:
        return 0
    return 1


def agmat(bed_file, inv=True, small_val=0.001, out_fmt='mat'):
    """
    additive genomic relationship matrix and its inversion
    :param bed_file: The prefix for plink binary file
    :param inv: Whether to calculate the inversion. Default value is True
    :param small_val: A small vale added to the diagonal to grant the positive definite. Default value is 0.001.
    :param out_fmt: the output format. mat: matrix format (default); row_col_val: row-column-value format;
    id_id_val: id-id-value format.
    :return: return numpy array for genomic relationship matrix and its inversion. Output the matrixes into the file
    with prefix of bed_file.
    """
    logging.info("{:#^80}".format("Read the SNP data"))
    snp_mat = read_plink(bed_file)
    if np.any(np.isnan(snp_mat)):
        logging.info('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    num_id = snp_mat.shape[0]  # 个体数
    num_snp = snp_mat.shape[1]  # SNP数
    logging.info("There are {:d} individuals and {:d} SNPs.".format(num_id, num_snp))
    freq = np.sum(snp_mat, axis=0) / (2 * num_id)
    freq.shape = (1, num_snp)
    scale = 2 * freq * (1 - freq)  # 标准化因子
    scale = np.sum(scale)
    logging.info('The scaled factor is: {:.3f}'.format(scale))
    snp_mat = snp_mat - 2 * freq
    
    logging.info("{:#^80}".format("Calculate the additive genomic relationship matrix"))
    clock_t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    kin = np.dot(snp_mat, snp_mat.T) / scale
    kin_diag = np.diag(kin)
    kin_diag = kin_diag + kin_diag * small_val
    np.fill_diagonal(kin, kin_diag)
    clock_t1 = time.perf_counter()
    cpu_t1 = time.process_time()
    logging.info("Running time: Clock time, {:.5f} sec; CPU time, {:.5f} sec.".format(clock_t1 - clock_t0, cpu_t1 - cpu_t0))
    
    logging.info("{:#^80}".format("Output"))
    fam_info = pd.read_csv(bed_file + '.fam', sep='\s+', header=None)
    id = np.array(fam_info.iloc[:, 1])
    out_file = bed_file + '.agrm'
    logging.info("The output file is " + out_file)
    res = output_mat(kin, id, out_file, out_fmt)
    if res is 0:
        logging.error('Not Recognized output format: ' + out_fmt)
        sys.exit()
    kin_inv = None
    if inv:
        logging.info("{:#^80}".format("Calculate the inversion of kinship"))
        clock_t0 = time.perf_counter()
        cpu_t0 = time.process_time()
        kin_inv = linalg.inv(kin)
        clock_t1 = time.perf_counter()
        cpu_t1 = time.process_time()
        logging.info("Running time: Clock time, {:.5f} sec; CPU time, {:.5f} sec.".format(clock_t1 - clock_t0,
                                                                                   cpu_t1 - cpu_t0))
        logging.info("{:#^80}".format("Output the inversion"))
        out_file = bed_file + '.agiv'
        logging.info("The output file is: " + out_file)
        output_mat(kin_inv, id, out_file, out_fmt)
    return kin, kin_inv


def dgmat_as(bed_file, inv=True, small_val=0.001, out_fmt='mat'):
    """
    dominance genomic relationship matrix and its inversion
    :param bed_file: The prefix for plink binary file
    :param inv: Whether to calculate the inversion. Default value is True
    :param small_val: A small vale added to the diagonal to grant the positive definite. Default value is 0.001.
    :param out_fmt: the output format. mat: matrix format (default); row_col_val: row-column-value format;
    id_id_val: id-id-value format.
    :return: return numpy array for genomic relationship matrix and its inversion. Output the matrixes into the file
    with prefix of bed_file.
    """
    logging.info("{:#^80}".format("Read, impute and scale the SNP"))
    snp_mat = read_plink(bed_file)
    if np.any(np.isnan(snp_mat)):
        print('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    num_id = snp_mat.shape[0]  # 个体数
    num_snp = snp_mat.shape[1]  # SNP数
    logging.info("There are {:d} individuals and {:d} SNPs.".format(num_id, num_snp))
    freq = np.sum(snp_mat, axis=0) / (2 * snp_mat.shape[0])
    freq.shape = (1, num_snp)
    scale_vec = 2 * freq * (1 - freq)
    scale = np.sum(scale_vec * (1 - scale_vec))
    logging.info('The scaled factor is: {:.3f}'.format(scale))
    snp_mat[snp_mat > 1.5] = 0.0  # 2替换为0, 变为0、1、0编码
    snp_mat = snp_mat - scale_vec

    logging.info("{:#^80}".format("Calculate the dominance genomic relationship matrix"))
    clock_t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    kin = np.dot(snp_mat, snp_mat.T) / scale
    kin_diag = np.diag(kin)
    kin_diag = kin_diag + kin_diag * small_val
    np.fill_diagonal(kin, kin_diag)
    clock_t1 = time.perf_counter()
    cpu_t1 = time.process_time()
    logging.info(
        "Running time: Clock time, {:.5f} sec; CPU time, {:.5f} sec.".format(clock_t1 - clock_t0, cpu_t1 - cpu_t0))

    logging.info("{:#^80}".format("Output"))
    fam_info = pd.read_csv(bed_file + '.fam', sep='\s+', header=None)
    id = np.array(fam_info.iloc[:, 1])
    out_file = bed_file + '.dgrm_as'
    logging.info("The output file is " + out_file)
    res = output_mat(kin, id, out_file, out_fmt)
    if res is 0:
        logging.error('Not Recognized output format: ' + out_fmt)
        sys.exit()
    kin_inv = None
    if inv:
        logging.info("{:#^80}".format("Calculate the inversion of kinship"))
        clock_t0 = time.perf_counter()
        cpu_t0 = time.process_time()
        kin_inv = linalg.inv(kin)
        clock_t1 = time.perf_counter()
        cpu_t1 = time.process_time()
        logging.info("Running time: Clock time, {:.5f} sec; CPU time, {:.5f} sec.".format(clock_t1 - clock_t0,
                                                                                   cpu_t1 - cpu_t0))
        logging.info("{:#^80}".format("Output the inversion"))
        out_file = bed_file + '.dgiv_as'
        logging.info("The output file is: " + out_file)
        output_mat(kin_inv, id, out_file, out_fmt)
    return kin, kin_inv


def ginbreedcoef(bed_file):
    snp_mat = read_plink(bed_file)
    if np.any(np.isnan(snp_mat)):
        print('Missing genotypes are imputed with random genotypes.')
        snp_mat = impute_geno(snp_mat)
    print("There are {:d} individuals and {:d} SNPs.".format(snp_mat.shape[0], snp_mat.shape[1]))
    homo_f = 1 - np.sum(np.abs(snp_mat - 1.0) < 0.01, axis=1) / snp_mat.shape[1]
    freq = np.sum(snp_mat, axis=0) / (2 * snp_mat.shape[0])
    freq.shape = (1, snp_mat.shape[1])
    scale_vec = 2 * freq * (1 - freq)
    scale = np.sum(scale_vec)
    snp_mat = snp_mat - 2 * freq
    grm_f1 = np.sum(np.multiply(snp_mat, snp_mat), axis=1) / scale - 1.0
    grm_f2 = np.sum(np.multiply(snp_mat, snp_mat) / scale_vec, axis=1) / snp_mat.shape[1] - 1.0
    
    fam_info = pd.read_csv(bed_file + '.fam', sep='\s+', header=None)
    id = np.array(fam_info.iloc[:, 1])
    data_df = {'id': id, 'homo_F': homo_f, 'grm_F1': grm_f1, 'grm_F2': grm_f2}
    data_df = pd.DataFrame(data_df, columns=['id', 'homo_F', 'grm_F1', 'grm_F2'])
    out_file = bed_file + '.ginbreedcoef'
    data_df.to_csv(out_file, sep=' ', header=True, index=False)


import logging
import gc
import numpy as np
import pandas as pd
from gmat.process_plink.process_plink import read_plink, impute_geno


def simu_epistasis_freq(bed_file, add_file, dom_file, epiAA_file, epiAD_file, epiDD_file, ratio=None, mean=1.0, res_var=1.0, out_file='simu_epistasis_freq'):
    logging.info("Read the SNP")
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
    if ratio is None:
        ratio = np.array([2.0, 1.0, 0.5, 0.5, 0.5, 1.0])
    else:
        ratio = np.array(ratio)
    logging.info("Additive")
    add_snp = pd.read_csv(add_file, header=None, sep='\s+')
    add_var = ratio[0]/ratio[-1]*res_var
    freq_a = freq[0, add_snp.iloc[:, 0]]
    add_snp_eff_var = 2*freq_a*(1-freq_a)*np.array(add_snp.iloc[:, 1])*np.array(add_snp.iloc[:, 1])
    add_snp.iloc[:, 1] = add_snp.iloc[:, 1]/np.sqrt(np.sum(add_snp_eff_var)/add_var)
    add_snp.to_csv(add_file + '.norm', sep=' ', header=False, index=False)
    logging.info("Dominance")
    dom_snp = pd.read_csv(dom_file, header=None, sep='\s+')
    dom_var = ratio[1] / ratio[-1] * res_var
    freq_d = freq[0, dom_snp.iloc[:, 0]]
    dom_snp_eff_var = 2 * freq_d * (1 - freq_d) * (1 - 2 * freq_d * (1 - freq_d)) * np.array(dom_snp.iloc[:, 1]) * np.array(dom_snp.iloc[:, 1])
    dom_snp.iloc[:, 1] = dom_snp.iloc[:, 1] / np.sqrt(np.sum(dom_snp_eff_var)/dom_var)
    dom_snp.to_csv(dom_file + '.norm', sep=' ', header=False, index=False)
    logging.info("Additive by additive epistasis")
    epiAA_snp = pd.read_csv(epiAA_file, header=None, sep='\s+')
    epiAA_var = ratio[2] / ratio[-1] * res_var
    epiAA_snp_eff_var = np.var(snp_matA[:, epiAA_snp.iloc[:, 0]] * snp_matA[:, epiAA_snp.iloc[:, 1]] * np.array(epiAA_snp.iloc[:, 2]), axis=0)
    epiAA_snp.iloc[:, 2] = epiAA_snp.iloc[:, 2]/np.sqrt(np.sum(epiAA_snp_eff_var)/epiAA_var)
    epiAA_snp.to_csv(epiAA_file + '.norm', sep=' ', header=False, index=False)
    logging.info("Additive by dominance epistasis")
    epiAD_snp = pd.read_csv(epiAD_file, header=None, sep='\s+')
    epiAD_var = ratio[3] / ratio[-1] * res_var
    epiAD_snp_eff_var = np.var(
        snp_matA[:, epiAD_snp.iloc[:, 0]] * snp_matD[:, epiAD_snp.iloc[:, 1]] * np.array(epiAD_snp.iloc[:, 2]), axis=0)
    epiAD_snp.iloc[:, 2] = epiAD_snp.iloc[:, 2] / np.sqrt(np.sum(epiAD_snp_eff_var)/epiAD_var)
    epiAD_snp.to_csv(epiAD_file + '.norm', sep=' ', header=False, index=False)
    logging.info("Dominance by dominance epistasis")
    epiDD_snp = pd.read_csv(epiDD_file, header=None, sep='\s+')
    epiDD_var = ratio[3] / ratio[-1] * res_var
    epiDD_snp_eff_var = np.var(
        snp_matD[:, epiDD_snp.iloc[:, 0]] * snp_matD[:, epiDD_snp.iloc[:, 1]] * np.array(epiDD_snp.iloc[:, 2]), axis=0)
    epiDD_snp.iloc[:, 2] = epiDD_snp.iloc[:, 2] / np.sqrt(np.sum(epiDD_snp_eff_var)/epiDD_var)
    epiDD_snp.to_csv(epiDD_file + '.norm', sep=' ', header=False, index=False)
    logging.info("Residual")
    res_vec = np.random.normal(0, np.sqrt(res_var), snp_matA.shape[0])
    np.savetxt(out_file + '.res', res_vec)
    logging.info("Phenotypic values")
    pheno_vec = mean + np.sum(snp_matA[:, add_snp.iloc[:, 0]] * np.array(add_snp.iloc[:, 1]), axis=1) + \
                np.sum(snp_matD[:, dom_snp.iloc[:, 0]] * np.array(dom_snp.iloc[:, 1]), axis=1) + \
    np.sum(snp_matA[:, epiAA_snp.iloc[:, 0]] * snp_matA[:, epiAA_snp.iloc[:, 1]] * np.array(epiAA_snp.iloc[:, 2]), axis=1) + \
    np.sum(snp_matA[:, epiAD_snp.iloc[:, 0]] * snp_matD[:, epiAD_snp.iloc[:, 1]] * np.array(epiAD_snp.iloc[:, 2]), axis=1) + \
    np.sum(snp_matD[:, epiDD_snp.iloc[:, 0]] * snp_matD[:, epiDD_snp.iloc[:, 1]] * np.array(epiDD_snp.iloc[:, 2]), axis=1) + \
    res_vec
    fam_df = pd.read_csv(bed_file + '.fam', sep='\s+', header=None)
    fam_df.iloc[:, 2] = 1
    fam_df.iloc[:, 3] = pheno_vec
    res_df = fam_df.iloc[:, 0:4]
    res_df.to_csv(out_file + '.pheno', sep=' ', header=False, index=False)
    return res_df


def simu_epistasis(bed_file, add_file, dom_file, epiAA_file, epiAD_file, epiDD_file, ratio=None, mean=1.0, res_var=1.0, out_file='simu_epistasis'):
    logging.info("Read the SNP")
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
    if ratio is None:
        ratio = np.array([2.0, 1.0, 0.5, 0.5, 0.5, 1.0])
    else:
        ratio = np.array(ratio)
    logging.info("Additive")
    add_snp = pd.read_csv(add_file, header=None, sep='\s+')
    add_var = ratio[0]/ratio[-1]*res_var
    add_snp_eff_var = np.var(snp_matA[:, add_snp.iloc[:, 0]] * np.array(add_snp.iloc[:, 1]), axis=0)
    add_snp.iloc[:, 1] = add_snp.iloc[:, 1]/np.sqrt(np.sum(add_snp_eff_var)/add_var)
    add_snp.to_csv(add_file + '.norm', sep=' ', header=False, index=False)
    logging.info("Dominance")
    dom_snp = pd.read_csv(dom_file, header=None, sep='\s+')
    dom_var = ratio[1] / ratio[-1] * res_var
    dom_snp_eff_var = np.var(snp_matD[:, dom_snp.iloc[:, 0]] * np.array(dom_snp.iloc[:, 1]), axis=0)
    dom_snp.iloc[:, 1] = dom_snp.iloc[:, 1] / np.sqrt(np.sum(dom_snp_eff_var)/dom_var)
    dom_snp.to_csv(dom_file + '.norm', sep=' ', header=False, index=False)
    logging.info("Additive by additive epistasis")
    epiAA_snp = pd.read_csv(epiAA_file, header=None, sep='\s+')
    epiAA_var = ratio[2] / ratio[-1] * res_var
    epiAA_snp_eff_var = np.var(snp_matA[:, epiAA_snp.iloc[:, 0]] * snp_matA[:, epiAA_snp.iloc[:, 1]] * np.array(epiAA_snp.iloc[:, 2]), axis=0)
    epiAA_snp.iloc[:, 2] = epiAA_snp.iloc[:, 2]/np.sqrt(np.sum(epiAA_snp_eff_var)/epiAA_var)
    epiAA_snp.to_csv(epiAA_file + '.norm', sep=' ', header=False, index=False)
    logging.info("Additive by dominance epistasis")
    epiAD_snp = pd.read_csv(epiAD_file, header=None, sep='\s+')
    epiAD_var = ratio[3] / ratio[-1] * res_var
    epiAD_snp_eff_var = np.var(
        snp_matA[:, epiAD_snp.iloc[:, 0]] * snp_matD[:, epiAD_snp.iloc[:, 1]] * np.array(epiAD_snp.iloc[:, 2]), axis=0)
    epiAD_snp.iloc[:, 2] = epiAD_snp.iloc[:, 2] / np.sqrt(np.sum(epiAD_snp_eff_var)/epiAD_var)
    epiAD_snp.to_csv(epiAD_file + '.norm', sep=' ', header=False, index=False)
    logging.info("Dominance by dominance epistasis")
    epiDD_snp = pd.read_csv(epiDD_file, header=None, sep='\s+')
    epiDD_var = ratio[3] / ratio[-1] * res_var
    epiDD_snp_eff_var = np.var(
        snp_matD[:, epiDD_snp.iloc[:, 0]] * snp_matD[:, epiDD_snp.iloc[:, 1]] * np.array(epiDD_snp.iloc[:, 2]), axis=0)
    epiDD_snp.iloc[:, 2] = epiDD_snp.iloc[:, 2] / np.sqrt(np.sum(epiDD_snp_eff_var)/epiDD_var)
    epiDD_snp.to_csv(epiDD_file + '.norm', sep=' ', header=False, index=False)
    logging.info("Residual")
    res_vec = np.random.normal(0, np.sqrt(res_var), snp_matA.shape[0])
    np.savetxt(out_file + '.res', res_vec)
    logging.info("Phenotypic values")
    pheno_vec = mean + np.sum(snp_matA[:, add_snp.iloc[:, 0]] * np.array(add_snp.iloc[:, 1]), axis=1) + \
                np.sum(snp_matD[:, dom_snp.iloc[:, 0]] * np.array(dom_snp.iloc[:, 1]), axis=1) + \
    np.sum(snp_matA[:, epiAA_snp.iloc[:, 0]] * snp_matA[:, epiAA_snp.iloc[:, 1]] * np.array(epiAA_snp.iloc[:, 2]), axis=1) + \
    np.sum(snp_matA[:, epiAD_snp.iloc[:, 0]] * snp_matD[:, epiAD_snp.iloc[:, 1]] * np.array(epiAD_snp.iloc[:, 2]), axis=1) + \
    np.sum(snp_matD[:, epiDD_snp.iloc[:, 0]] * snp_matD[:, epiDD_snp.iloc[:, 1]] * np.array(epiDD_snp.iloc[:, 2]), axis=1) + \
    res_vec
    fam_df = pd.read_csv(bed_file + '.fam', sep='\s+', header=None)
    fam_df.iloc[:, 2] = 1
    fam_df.iloc[:, 3] = pheno_vec
    res_df = fam_df.iloc[:, 0:4]
    res_df.to_csv(out_file + '.pheno', sep=' ', header=False, index=False)
    return res_df

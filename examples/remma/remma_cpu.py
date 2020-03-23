"""
additive by additive test by SNP-BLUP model. Mouse data.

"""
import logging
import numpy as np
from glta.uvlmm import wemai_multi_gmat
from gmat.gmatrix import agmat, dgmat_as
from gmat.remma.remma_cpu import remma_add_cpu

logging.basicConfig(level=logging.INFO)

# prepare the phenotypic vector, design matrixed for fixed effects and random effects
from gmat.uvlmm import design_matrix_wemai_multi_gmat

pheno_file = '../data/mouse/pheno'
bed_file = '../data/mouse/plink'
y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)

# Calculate the genomic relationship matrix
from gmat.gmatrix.gmatrix import agmat, dgmat_as
a = agmat(bed_file, inv=False)
b = dgmat_as(bed_file, inv=False)


# Example 1: Just include additive relationship matrix
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
gmat_lst = [a[0]]
var_com_a = wemai_multi_gmat(y, xmat, zmat, gmat_lst)
res_a = remma_add_cpu(y, xmat, zmat, gmat_lst, var_com_a, bed_file, out_file='../data/mouse/remma_add_cpu_a')


# Example 2: Include additive relationship matrix and dominance relationship matrix
gmat_lst = [a[0], b[0]]
var_com_a_d = wemai_multi_gmat(y, xmat, zmat, gmat_lst)
res_a_d = remma_add_cpu(y, xmat, zmat, gmat_lst, var_com_a_d, bed_file, out_file='../data/mouse/remma_add_cpu_a_d')


# Example 3: Include additive relationship matrix and additive by additive relationship matrix
gmat_lst = [a[0], a[0]*a[0]]
var_com_a_axa = wemai_multi_gmat(y, xmat, zmat, gmat_lst)
res_a_axa = remma_add_cpu(y, xmat, zmat, gmat_lst, var_com_a_axa, bed_file, out_file='../data/mouse/remma_add_cpu_a_axa')


# Example 4: Include additive, dominance, additive by additive, additive by dominance,
# dominance by dominance relationship matrix
gmat_lst = [a[0], b[0], a[0]*a[0], a[0]*b[0], b[0]*b[0]]
var_com_all = wemai_multi_gmat(y, xmat, zmat, gmat_lst)
res_all = remma_add_cpu(y, xmat, zmat, gmat_lst, var_com_all, bed_file, out_file='../data/mouse/remma_add_cpu_all')

print('Median values: ', np.median(res_a['p_val']), np.median(res_a_d['p_val']), np.median(res_a_axa['p_val']),
      np.median(res_all['p_val']))
print('Pearson product-moment correlation coefficients: ', np.corrcoef([res_a['p_val'], res_a_d['p_val'], res_a_axa['p_val'], res_all['p_val']]))





import pandas as pd
import numpy as np
from scipy.stats import chi2
import logging
from glta.gmat import agmat, dgmat_as
from glta.uvlmm import wemai_multi_gmat

from glta.remma.remma_cpu import remma_add_cpu
from glta.remma.remma_cpu import remma_dom_cpu

from glta.remma.remma_cpu import remma_epiAA_cpu, remma_epiAA_select_cpu, remma_epiAA_pair_cpu, \
    remma_epiAA_eff_cpu, remma_epiAA_eff_cpu_c

from glta.remma.remma_cpu import remma_epiAD_cpu, remma_epiAD_select_cpu, remma_epiAD_pair_cpu, \
    remma_epiAD_eff_cpu, remma_epiAD_eff_cpu_c

from glta.remma.remma_cpu import remma_epiDD_cpu, remma_epiDD_select_cpu, remma_epiDD_pair_cpu, \
    remma_epiDD_eff_cpu, remma_epiDD_eff_cpu_c




logging.basicConfig(level=logging.INFO)



bed_file = '../data/plink'
df = pd.read_csv("../data/pheno", header=None, sep='\s+')

y = np.array(df.iloc[:, -1:])
xmat = np.array(df.iloc[:, 2:-1])



agmat_lst = agmat(bed_file, inv=True, small_val=0.001, out_fmt='mat')
dgmat_lst = dgmat_as(bed_file, inv=True, small_val=0.001, out_fmt='mat')


gmat_lst = [agmat_lst[0], dgmat_lst[0], agmat_lst[0]*agmat_lst[0], agmat_lst[0]*dgmat_lst[0], dgmat_lst[0]*dgmat_lst[0]]

var_com = wemai_multi_gmat(y, xmat, gmat_lst, init=None, maxiter=200, cc=1.0e-8)


res_add = remma_add_cpu(y, xmat, gmat_lst, var_com, bed_file, out_file='remma_add_cpu')
print(np.median(res_add['p_val']))

res_dom = remma_dom_cpu(y, xmat, gmat_lst, var_com, bed_file, out_file='remma_dom_cpu')
print(np.median(res_dom['p_val']))

###########加加互作
res_epiAA_cpu = remma_epiAA_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, p_cut=1, out_file='remma_epiAA_cpu')
p_vec = chi2.sf(res_epiAA_cpu[:, -1], 1)
print(np.median(p_vec))

res_epiAA_select_cpu = remma_epiAA_select_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, snp_lst_1=None, p_cut=1.0, out_file='remma_epiAA_select_cpu')
p_vec = chi2.sf(res_epiAA_select_cpu[:, -1], 1)
print(np.median(p_vec))


snp1 = np.arange(1, 1407).reshape(-1, 1)
snp2 = np.array([0]*1406).reshape(-1, 1)
snp_pair = np.concatenate([snp2, snp1], axis=1)
res_epiAA_pair_cpu = remma_epiAA_pair_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_pair, out_file='remma_epiAA_pair_cpu')
p_vec = chi2.sf(res_epiAA_pair_cpu[:, -1], 1)
print(np.median(p_vec))

res_epiAA_eff_cpu = remma_epiAA_eff_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, eff_cut=-999.0, out_file='remma_epiAA_eff_cpu')

res_epiAA_eff_cpu_c = remma_epiAA_eff_cpu_c(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, eff_cut=-999.0, out_file='remma_epiAA_eff_cpu_c')


###########加显互作
res_epiAD_cpu = remma_epiAD_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, p_cut=1, out_file='remma_epiAD_cpu')
p_vec = chi2.sf(res_epiAD_cpu[:, -1], 1)
print(np.median(p_vec))

res_epiAD_select_cpu = remma_epiAD_select_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, snp_lst_1=None, p_cut=1.0, out_file='remma_epiAD_select_cpu')
p_vec = chi2.sf(res_epiAD_select_cpu[:, -1], 1)
print(np.median(p_vec))


snp1 = np.arange(1, 1407).reshape(-1, 1)
snp2 = np.array([0]*1406).reshape(-1, 1)
snp_pair = np.concatenate([snp2, snp1], axis=1)
res_epiAD_pair_cpu = remma_epiAD_pair_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_pair, out_file='remma_epiAD_pair_cpu')
p_vec = chi2.sf(res_epiAD_pair_cpu[:, -1], 1)
print(np.median(p_vec))

res_epiAD_eff_cpu = remma_epiAD_eff_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, eff_cut=-999.0, out_file='remma_epiAD_eff_cpu')

res_epiAD_eff_cpu_c = remma_epiAD_eff_cpu_c(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, eff_cut=-999.0, out_file='remma_epiAD_eff_cpu_c')

###########显显互作
res_epiDD_cpu = remma_epiDD_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, p_cut=1, out_file='remma_epiDD_cpu')
p_vec = chi2.sf(res_epiDD_cpu[:, -1], 1)
print(np.median(p_vec))

res_epiDD_select_cpu = remma_epiDD_select_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, snp_lst_1=None, p_cut=1.0, out_file='remma_epiDD_select_cpu')
p_vec = chi2.sf(res_epiDD_select_cpu[:, -1], 1)
print(np.median(p_vec))


snp1 = np.arange(1, 1407).reshape(-1, 1)
snp2 = np.array([0]*1406).reshape(-1, 1)
snp_pair = np.concatenate([snp2, snp1], axis=1)
res_epiDD_pair_cpu = remma_epiDD_pair_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_pair, out_file='remma_epiDD_pair_cpu')
p_vec = chi2.sf(res_epiDD_pair_cpu[:, -1], 1)
print(np.median(p_vec))

res_epiDD_eff_cpu = remma_epiDD_eff_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, eff_cut=-999.0, out_file='remma_epiDD_eff_cpu')

res_epiDD_eff_cpu_c = remma_epiDD_eff_cpu_c(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, eff_cut=-999.0, out_file='remma_epiDD_eff_cpu_c')







var_com = [0.06289206, 0.07641075, 0.08121168]
# res_epiAA_eff_cpu = remma_epiAA_eff_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, eff_cut=-999.0, out_file='remma_epiAA_eff_cpu')

var_com = [0.06289206, 0.07641075, 0.08121168]
res_epiAA_eff_cpu_c = remma_epiAA_eff_cpu_c(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, eff_cut=-999.0, out_file='remma_epiAA_eff_cpu_c')

'''
res_epiAA_select_cpu = remma_epiAA_select_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_lst_0=None, snp_lst_1=None, p_cut=1.0, out_file='remma_epiAA_select_cpu')


snp1 = np.arange(1407).reshape(-1, 1)
snp2 = np.array([0]*1407).reshape(-1, 1)
snp_pair = np.concatenate([snp2, snp1], axis=1)
res_epiAA_pair_cpu = remma_epiAA_pair_cpu(y, xmat, gmat_lst, var_com, bed_file, snp_pair, out_file='remma_epiAA_pair_cpu')


from glta.remma import remma_random_pair

num_snp = 1000
remma_random_pair(num_snp, num_pair=100000, out_file='random_pair')


'''

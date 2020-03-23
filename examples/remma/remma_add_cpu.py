"""
additive by additive test by SNP-BLUP model. Mouse data.

"""
import logging
import numpy as np
from gmat.uvlmm import design_matrix_wemai_multi_gmat
from gmat.uvlmm import wemai_multi_gmat
from gmat.gmatrix import agmat, dgmat_as
from gmat.remma.remma_cpu import remma_add_cpu

logging.basicConfig(level=logging.INFO)

# prepare the phenotypic vector, design matrixed for fixed effects and random effects
pheno_file = '../data/mouse/pheno'
bed_file = '../data/mouse/plink'
y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)

# Calculate the genomic relationship matrix
a = agmat(bed_file, inv=False)
b = dgmat_as(bed_file, inv=False)


# Example 1: Just include additive relationship matrix
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


"""
additive by additive test by SNP-BLUP model. Yeast data. No heterozygous genotype. No dominance. 
Include repeated measures.

"""

import logging
import numpy as np
from gmat.uvlmm import design_matrix_wemai_multi_gmat
from gmat.uvlmm import wemai_multi_gmat
from gmat.gmatrix import agmat, dgmat_as
from gmat.remma.remma_cpu import remma_add_cpu

logging.basicConfig(level=logging.INFO)

# prepare the phenotypic vector, design matrixed for fixed effects and random effects
pheno_file = '../data/yeast/CobaltChloride'
bed_file = '../data/yeast/CobaltChloride'
y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)

# Calculate the genomic relationship matrix
a = agmat(bed_file, inv=False)

# Example 0: Include additive relationship matrix
gmat_lst = [a[0]]  # np.eye(a[0].shape[0]) is a identity matrix to model the permanent environmental effect
var_com_a = wemai_multi_gmat(y, xmat, zmat, gmat_lst)
res_a = remma_add_cpu(y, xmat, zmat, gmat_lst, var_com_a, bed_file, out_file='../data/yeast/remma_add_cpu_a')


# Example 1: Include additive relationship matrix and individual-specific errors (Permanent environmental effect)
gmat_lst = [a[0], np.eye(a[0].shape[0])]  # np.eye(a[0].shape[0]) is a identity matrix to model the permanent environmental effect
var_com_a_p = wemai_multi_gmat(y, xmat, zmat, gmat_lst)
res_a_p = remma_add_cpu(y, xmat, zmat, gmat_lst, var_com_a_p, bed_file, out_file='../data/yeast/remma_add_cpu_a_p')


# Example 2: Include additive, additive by additive relationship matrix
# and individual-specific errors (Permanent environmental effect)

gmat_lst = [a[0], a[0]*a[0], np.eye(a[0].shape[0])]  # np.eye(a[0].shape[0]) is a identity matrix to model the permanent environmental effect
var_com_a_axa_p = wemai_multi_gmat(y, xmat, zmat, gmat_lst)
res_a_axa_p = remma_add_cpu(y, xmat, zmat, gmat_lst, var_com_a_axa_p, bed_file,
                            out_file='../data/yeast/remma_add_cpu_a_axa_p')

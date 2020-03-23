"""
Analyze the mouse data. Partition the phenotypic variance into addtive + dominance + additve by additive + additive by dominance
+ dominance by dominance + residual

"""
import logging
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


# Estimate the variances
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
gmat_lst = [a[0], b[0], a[0]*a[0], a[0]*b[0],  b[0]*b[0]]
res = wemai_multi_gmat(y, xmat, zmat, gmat_lst, init=None, maxiter=200, cc_par=1.0e-8, cc_gra=1.0e-6)



"""
Analyze the yeast data. Partition the phenotypic variance into addtive + dominance + additve by additive + additive by dominance
+ dominance by dominance + individual-specific residual + residual

"""
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

# prepare the phenotypic vector, design matrixed for fixed effects and random effects
from gmat.uvlmm import design_matrix_wemai_multi_gmat

pheno_file = '../data/yeast/CobaltChloride'
bed_file = '../data/yeast/CobaltChloride'
y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)

# Calculate the genomic relationship matrix
from gmat.gmatrix.gmatrix import agmat
a = agmat(bed_file, inv=False)

# Estimate the variances
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
gmat_lst = [a[0], a[0]*a[0], np.eye(a[0].shape[0])]
res = wemai_multi_gmat(y, xmat, zmat, gmat_lst, init=None, maxiter=200, cc_par=1.0e-8, cc_gra=1.0e-6)




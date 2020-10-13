import logging
import sys
import numpy as np
from scipy.sparse import csr_matrix


def design_matrix_wemai_multi_gmat(pheno_file, bed_file):
    """
    prepare the phenotypic vectors, design matrixes of fixed effect and random effect for wemai_multi_gmat program
    :param pheno_file: phenotypic file. The fist two columns are family id, individual id which are same as plink *.fam
    file. The third column is always ones for population mean. The last column is phenotypic values. The ohter covariate
    can be added between columns for population mean and phenotypic values.
    :param bed_file: the prefix for binary file
    :return: phenotypic vectors, design matrixes of fixed effect, design matrixes of random effect (csr_matrix form)
    """
    id_bed_lst = []
    with open(bed_file + '.fam') as fin:
        for line in fin:
            arr = line.split()
            id_bed_lst.append(" ".join([arr[0], arr[1]]))
    id_pheno_lst = {}
    with open(pheno_file) as fin:
        for line in fin:
            arr = line.split()
            if arr[-1] not in ['NA', 'NaN', 'nan', 'na']:
                try:
                    id_pheno_lst[" ".join([arr[0], arr[1]])].append(" ".join(arr))
                except Exception as e:
                    del e
                    id_pheno_lst[" ".join([arr[0], arr[1]])] = [" ".join(arr)]
    id_not_pheno = set(id_bed_lst) - set(list(id_pheno_lst.keys()))
    if len(id_not_pheno) > 0:
        logging.error('The below genotyped id is not in the phenotype file:\n {}'.format('\n'.join(list(id_not_pheno))))
        sys.exit()
    y = []
    xmat = []
    id_lst = []
    for id in id_bed_lst:
        for val in id_pheno_lst[id]:
            arr = val.split()
            y.append(float(arr[-1]))
            xmat.append(arr[2:-1])
            id_lst.append(arr[1])
    y = np.array(y).reshape(-1, 1)
    xmat = np.array(xmat, dtype=float).reshape(y.shape[0], -1)
    id_dct = {}
    row = []
    col = []
    j = 0
    for i in range(len(id_lst)):
        row.append(i)
        if id_lst[i] not in id_dct:
            id_dct[id_lst[i]] = j
            j += 1
        col.append(id_dct[id_lst[i]])
    zmat = csr_matrix(([1.0]*len(row), (row, col)))
    return y, xmat, zmat


def design_matrix_wemai_multi_gmat_pred(pheno_file, bed_file):
    """
    prepare the phenotypic vectors, design matrixes of fixed effect and random effect for wemai_multi_gmat program
    :param pheno_file: phenotypic file. The fist two columns are family id, individual id which are same as plink *.fam
    file. The third column is always ones for population mean. The last column is phenotypic values. The ohter covariate
    can be added between columns for population mean and phenotypic values.
    :param bed_file: the prefix for binary file
    :return: phenotypic vectors, design matrixes of fixed effect, design matrixes of random effect (csr_matrix form)
    """
    id_bed_lst = []
    with open(bed_file + '.fam') as fin:
        for line in fin:
            arr = line.split()
            id_bed_lst.append(" ".join([arr[0], arr[1]]))
    id_pheno_lst = {}
    with open(pheno_file) as fin:
        for line in fin:
            arr = line.split()
            if arr[-1] not in ['NA', 'NaN', 'nan', 'na']:
                try:
                    id_pheno_lst[" ".join([arr[0], arr[1]])].append(" ".join(arr))
                except Exception as e:
                    del e
                    id_pheno_lst[" ".join([arr[0], arr[1]])] = [" ".join(arr)]
    y = []
    xmat = []
    id_lst = []
    for id in id_bed_lst:
        if id in id_pheno_lst:
            for val in id_pheno_lst[id]:
                arr = val.split()
                y.append(float(arr[-1]))
                xmat.append(arr[2:-1])
                id_lst.append(arr[1])
        else:
            id_lst.append('NA')
    y = np.array(y).reshape(-1, 1)
    xmat = np.array(xmat, dtype=float).reshape(y.shape[0], -1)
    id_dct = {}
    row, col, val = [], [], []
    rowi, coli = 0, 0
    for i in range(len(id_lst)):
        if id_lst[i] != 'NA':
            row.append(rowi)
            rowi += 1
            if id_lst[i] not in id_dct:
                id_dct[id_lst[i]] = coli
                coli += 1
            col.append(id_dct[id_lst[i]])
            val.append(1.0)
        else:
            coli += 1
    zmat = csr_matrix((val, (row, col)), shape=(rowi, coli))
    return y, xmat, zmat

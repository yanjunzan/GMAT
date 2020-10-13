from gmat.longwas.unbalance.unbalance_varcom import unbalance_varcom

from scipy import sparse
from scipy.sparse import csr_matrix, hstack
import numpy as np
import pandas as pd
from patsy import dmatrix
import gc
import os

def leg_mt(time, tmax, tmin, order):
    time = np.array(time, dtype=float).reshape(max(time.shape), 1)
    tvec = 2 * (time - tmin) / (tmax - tmin) - 1
    pmat = []
    for k in range(order + 1):
        c = int(k / 2)
        j = k
        p = 0
        for r in range(0, c + 1):
            p += np.sqrt((2 * j + 1.0) / 2.0) * pow(0.5, j) * (pow(-1, r) *
                                    np.math.factorial(2 * j - 2 * r) / (np.math.factorial(r) *
                                    np.math.factorial(j - r) * np.math.factorial(j - 2 * r))) * pow(tvec, j - 2 * r)
        p = np.array(p)
        pmat.append(p)
    return pmat

from collections import defaultdict
def dct_2D():
    # declare the 2D dict
    return defaultdict(dct_21D)


def dct_21D():
    return defaultdict()



os.chdir("/data/zhanglab/ningc/Acode/GMAT/PKG/examples/data/mouse_long/")

import logging
from gmat.gmatrix import agmat, dgmat_as

logging.basicConfig(level=logging.INFO)

bed_file = 'plink'

# matrix form
agmat0 = agmat(bed_file, inv=True, small_val=0.001, out_fmt='mat')



data_file = 'phe.unbalance.txt'
data_df = pd.read_csv(data_file, sep='\s+', header=0)

na_method = 'omit'
if na_method == 'omit':
    data_df = data_df.dropna()
elif na_method == 'include':
    data_df = data_df.fillna(method='ffill')
    data_df = data_df.fillna(method='bfill')

col_names = data_df.columns

class_vec = []
for val in col_names:
    if not val[0].isalpha():
        print("The first character of columns names must be alphabet!")
        exit()
    if val[0] == val.capitalize()[0]:
        class_vec.append(val)
        data_df[val] = data_df[val].astype('str')
    else:
        try:
            data_df[val] = data_df[val].astype('float')
        except Exception as e:
            print(e)
            print(val, "may contain string, please check!")
            exit()


id_order = []
id = 'ID'
id_arr = list(data_df[id])
id_order.append(id_arr[0])
for i in range(1, len(id_arr)):
    if id_arr[i] != id_arr[i - 1]:
        id_order.append(id_arr[i])

id_in_data = set(data_df[id])

tpoint = 'weak'
trait = 'trait'


code_val = {}
code_dct = dct_2D()
for val in class_vec:
    code_val[val] = 0
    temp = []
    for i in range(data_df.shape[0]):
        if data_df[val][i] not in code_dct[val]:
            code_val[val] += 1
            code_dct[val][data_df[val][i]] = str(code_val[val])
        temp.append(code_dct[val][data_df[val][i]])

    data_df[val] = np.array(temp)

for val in class_vec:
    data_df[val] = data_df[val].astype('int')

forder = 3

leg_fix = leg_mt(data_df[tpoint],1, 16,  forder)
xmat_t = np.concatenate(leg_fix, axis=1)
xmat_t = csr_matrix(xmat_t)


max_id = max(data_df[id]) + 1
tmin = min(data_df[tpoint])
tmax = max(data_df[tpoint])
leg_lst = []  # legendre polynomials for time dependent fixed SNP effects, save for each individuals
for i in range(1, max_id):
    leg_lst.append(leg_mt(data_df[data_df[id] == i][tpoint], tmax, tmin, forder))

aorder = 3
porder = 3

kin_inv = agmat0[1]
leg_add = leg_mt(data_df[tpoint], 1, 16, aorder)
row = np.array(range(data_df.shape[0]))
col = np.array(data_df[id]) - 1
val = np.array([1.0] * data_df.shape[0])
add_mat = csr_matrix((val, (row, col)), shape=(data_df.shape[0], kin_inv.shape[0]))
zmat_add = []
for i in range(len(leg_add)):
    zmat_add.append(add_mat.multiply(leg_add[i]))

leg_per = leg_mt(data_df[tpoint], 1, 16, porder)
per_mat = csr_matrix((val, (row, col)))
zmat_per = []
for i in range(len(leg_per)):
    zmat_per.append((per_mat.multiply(leg_per[i])))

zmat = [zmat_add, zmat_per]
y = data_df[trait].values.reshape(data_df.shape[0], 1)

unbalance_varcom(y, xmat_t, zmat, kin_inv, init=None, max_iter=100, cc_par=1e-08, cc_gra=1e-06, em_weight_step=0.01)

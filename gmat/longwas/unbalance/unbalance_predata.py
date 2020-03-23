import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
from gmat.longwas.common import dtype_in_datafile, check_id_order, recode_fac_lst, leg_mt, design_matrix


def unbalance_predata(data_file, id, tpoint, trait, gmat_inv_file, tfix=None, fix=None, forder=3, aorder=3,
                      porder=3, na_method='omit'):
    data_df = pd.DataFrame()
    try:
        data_df = pd.read_csv(data_file, sep='\s+', header=0)
    except Exception as e:
        print(e)
        print("Fail to open the data file.")
        exit()
    
    print('How to process missing data:', na_method)
    if na_method == 'omit':
        data_df = data_df.dropna()
    elif na_method == 'include':
        data_df = data_df.fillna(method='ffill')
        data_df = data_df.fillna(method='bfill')
    else:
        print('{} method does not exist'.format(na_method))
        exit()
    
    col_names = data_df.columns
    print('The column names of data file:', ' '.join(list(col_names)))
    data_df, class_vec = dtype_in_datafile(data_df)
    
    print('Individual column:', id)
    if id not in col_names:
        print(id, 'is not in the data file, please check!')
        exit()
    if id not in class_vec:
        print('The initial letter of', id, 'should be capital!')
        exit()
    id_order = check_id_order(data_df[id])
    
    print('Time points column:', tpoint)
    if tpoint not in col_names:
        print(tpoint, 'is not in the data file, please check!')
        exit()
    if tpoint in class_vec:
        print('The initial letter of', tpoint, 'should be lowercase')
        exit()
    
    print('Trait column: ', trait)
    if trait not in col_names:
        print(trait, 'is not in the data file, please check!')
        exit()
    if trait in class_vec:
        print('The initial letter of', trait, 'should be lowercase')
        exit()
    y = np.array(data_df[trait]).reshape(data_df.shape[0], 1)
    
    print('Recode factor variables with integer in the data file:', ' '.join(list(class_vec)))
    code_fac_dct = {}
    for val in class_vec:
        fac_lst_code, code_dct = recode_fac_lst(list(data_df[val]))
        code_fac_dct[val] = code_dct
        data_df[val] = np.array(fac_lst_code, dtype=np.int64)
    
    print('Time dependent fix effect:', str(tfix))
    print('Legendre order is:', forder)
    leg_fix = leg_mt(data_df[tpoint], max(data_df[tpoint]), min(data_df[tpoint]), forder)
    fix_pos_vec = [0]
    if tfix is None:
        xmat_t = np.concatenate(leg_fix, axis=1)
        xmat_t = csr_matrix(xmat_t)
        tfix_len = np.array([forder + 1])
    else:
        tfix_mat, tfix_len = design_matrix(tfix, class_vec, data_df)
        tfix_len = np.array(tfix_len) * (forder + 1)
        xmat_t = []
        for veci in range(len(tfix_mat)):
            for vecj in range(len(leg_fix)):
                xmat_t.append(tfix_mat[veci].multiply(leg_fix[vecj]))
        xmat_t = hstack(xmat_t)
    for i in list(tfix_len):
        fix_pos_vec.append(fix_pos_vec[-1] + i)
    
    print('Time independent fix effect:', str(fix))
    if fix is None:
        xmat_nt = None
    else:
        xmat_nt, fix_len = design_matrix(fix, class_vec, data_df)
        xmat_nt = hstack(xmat_nt[1:])
        for i in list(fix_len[1:]):
            fix_pos_vec.append(fix_pos_vec[-1] + i)
    xmat = hstack([xmat_t, xmat_nt])
    
    print('Read the inversion of kinship matrix')
    row = []
    col = []
    kin_inv = []
    id_in_kin = {}
    code_val = max(np.array(list(code_fac_dct[id].values())).astype(np.int))
    with open(gmat_inv_file, 'r') as fin:
        for line in fin:
            arr = line.split()
            id_in_kin[arr[0]] = 1
            id_in_kin[arr[1]] = 1
            if arr[0] not in code_fac_dct[id]:
                code_val += 1
                code_fac_dct[id][arr[0]] = str(code_val)
            if arr[1] not in code_fac_dct[id]:
                code_val += 1
                code_fac_dct[id][arr[1]] = str(code_val)
            row.append(int(code_fac_dct[id][arr[0]]))
            col.append(int(code_fac_dct[id][arr[1]]))
            kin_inv.append(float(arr[2]))
    kin_inv = csr_matrix((np.array(kin_inv), (np.array(row) - 1, np.array(col) - 1))).toarray()
    kin_inv = np.add(kin_inv, kin_inv.T)  # maybe not lower triangular matrix
    np.fill_diagonal(kin_inv, 0.5 * np.diag(kin_inv))
    id_in_kin = set(id_in_kin.keys())
    id_in_data = set(id_order)
    id_not_in_kin = list(id_in_data - id_in_kin)
    if len(id_not_in_kin) != 0:
        print('The ID:', id_not_in_kin, 'in the data file is not in the kinship file!')
        exit()
    
    print('Build the design matrix for random effect.')
    print('Legendre order for additive effects:', str(aorder))
    leg_add = leg_mt(data_df[tpoint], max(data_df[tpoint]), min(data_df[tpoint]), aorder)
    row = np.arange(data_df.shape[0])
    col = np.array(data_df[id]) - 1
    val = np.array([1.0] * data_df.shape[0])
    add_mat = csr_matrix((val, (row, col)), shape=(data_df.shape[0], kin_inv.shape[0]))
    zmat_add = []
    for i in range(len(leg_add)):
        zmat_add.append(add_mat.multiply(leg_add[i]))
    
    print('Legendre order for permanent environmental effect:', str(porder))
    leg_per = leg_mt(data_df[tpoint], max(data_df[tpoint]), min(data_df[tpoint]), porder)
    per_mat = csr_matrix((val, (row, col)))
    zmat_per = []
    for i in range(len(leg_per)):
        zmat_per.append((per_mat.multiply(leg_per[i])))
    zmat = [zmat_add, zmat_per]
    kin_inv = [kin_inv, sparse.eye(max(data_df[id]), format="csr")]
    
    return y, xmat, zmat, kin_inv, code_fac_dct[id], id_order, fix_pos_vec

import numpy as np
from scipy.sparse import csr_matrix


def dtype_in_datafile(data_df):
    print('Note: Variates beginning with a capital letter is converted into factors.')
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
    return data_df, class_vec


def check_id_order(id):
    id_order = []
    id_arr = list(id)
    id_order.append(id_arr[0])
    for i in range(1, len(id_arr)):
        if id_arr[i] != id_arr[i - 1]:
            id_order.append(id_arr[i])
    id_in_data = set(id)
    if len(id_in_data) - len(id_order) != 0:
        print('The data is not sored by individual ID!')
        exit()
    return id_order


def recode_fac_lst(fac_lst):
    code_val = 0
    code_dct = {}
    fac_lst_code = []
    for i in range(len(fac_lst)):
        if fac_lst[i] not in code_dct:
            code_val += 1
            code_dct[fac_lst[i]] = str(code_val)
        fac_lst_code.append(code_dct[fac_lst[i]])
    return fac_lst_code, code_dct


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


def design_matrix(formula, class_vec, data_df):
    formula_new = formula.replace('+', ' ')
    formula_vec = formula_new.split()
    dmat_lst = [csr_matrix(np.ones((data_df.shape[0], 1)))]
    dmat_len = [1]
    for vari in formula_vec:
        if vari not in data_df.columns:
            print(vari, 'not in the data file!')
            exit()
        if vari in class_vec:
            row = np.arange(data_df.shape[0])
            col = np.array(data_df[vari]) - 1
            val = np.ones(data_df.shape[0])
            dmat_lst.append(csr_matrix((val, (row, col)))[:, 1:])
        else:
            dmat_lst.append(csr_matrix(np.array(data_df[vari]).reshape(data_df.shape[0], 1)))
        dmat_len.append(dmat_lst[-1].shape[1])
    return dmat_lst, dmat_len

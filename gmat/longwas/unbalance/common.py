import numpy as np
from collections import defaultdict
def is_int(num):
    try:
        int(num)
        return True
    except ValueError:
        return False

def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

###declare the 2D dict
def dct_3D():
    return defaultdict(dct_31D)

def dct_31D():
    return defaultdict(dct_32D)

def dct_32D():
    return defaultdict()


###declare the 2D dict
def dct_2D():
    return defaultdict(dct_21D)

def dct_21D():
    return defaultdict()


###declare the 1D dict
def dct_1D():
    return defaultdict()

###triple matrix multiplication a x b x a'

def tri_matT(a, b):
    res = np.dot(a,b)
    res = np.dot(res,a.T)
    return res

###triple matrix multiplication a x b x c

def tri_mat(a, b, c):
    res = np.dot(a, b)
    res = np.dot(res, c)
    return res

###triple matrix multiplication a x D x a'  D is diagonal matrix in row vector form

def Dtri_matT(a, b):
    res = np.multiply(a,b)
    res = np.dot(res,a.T)
    return res

###triple matrix multiplication a x D x c  D is diagonal matrix in row vector form

def Dtri_mat(a, b, c):
    res = np.multiply(a,b)
    res = np.dot(res, c)
    return res

####legendre polynomial
def leg(time, order):
    time = np.array(time, dtype=float).reshape(max(time.shape), 1)
    tmin = min(time)
    tmax = max(time)
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
    pmat = np.concatenate(pmat, axis=1)
    return pmat


def cal_xvkvkvx(xmat, vinv, onei, onej, onei2, onej2, leg_tp, kin_dct, var_ind, m, n):
    temp1 = reduce(np.matmul, [onei.T, leg_tp.T, vinv, xmat])
    temp2 = reduce(np.matmul, [onej.T, leg_tp.T, vinv, leg_tp, onei2])
    temp3 = reduce(np.matmul, [onej2.T, leg_tp.T, vinv, xmat])
    xvkvkvx = reduce(np.matmul, [temp1.transpose(0, 2, 1), temp2, temp3])
    xvkvkvx = reduce(np.multiply, [kin_dct[var_ind[m, 0]], kin_dct[var_ind[n, 0]], xvkvkvx])
    xvkvkvx = reduce(np.add, xvkvkvx)
    return xvkvkvx

def cal_xvkvkvx_res(xmat, vinv, onei, onej, leg_tp, kin_dct, var_ind, m):
    temp1 = reduce(np.matmul, [onei.T, leg_tp.T, vinv, xmat])
    temp2 = reduce(np.matmul, [onej.T, leg_tp.T, vinv, vinv, xmat])
    xvkvkvx = reduce(np.matmul, [temp1.transpose(0, 2, 1), temp2])
    xvkvkvx = reduce(np.multiply, [kin_dct[var_ind[m, 0]], xvkvkvx])
    xvkvkvx = reduce(np.add, xvkvkvx)
    return xvkvkvx




def longwas_lm(y, xmat):
    n = y.shape[0]
    r = xmat.shape[1]
    xx = np.linalg.inv(np.dot(xmat.T, xmat))
    yy = np.dot(y.T, y)
    yx = np.dot(y.T, xmat)
    yxxy = reduce(np.dot, [yx, xx, yx.T])
    sigma = np.sum((yy - yxxy)/(n - r))
    eff = np.dot(xx, yx.T)
    eff_var = xx*sigma
    return eff, eff_var, sigma

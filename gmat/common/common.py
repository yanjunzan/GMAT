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



def dct_3D():
    # declare the 3D dict
    return defaultdict(dct_31D)


def dct_31D():
    return defaultdict(dct_32D)


def dct_32D():
    return defaultdict()



def dct_2D():
    # declare the 2D dict
    return defaultdict(dct_21D)


def dct_21D():
    return defaultdict()


def dct_1D():
    # declare the 1D dict
    return defaultdict()


def tri_matT(a, b):
    # triple matrix multiplication a x b x a'
    res = np.dot(a, b)
    res = np.dot(res, a.T)
    return res


def tri_mat(a, b, c):
    # triple matrix multiplication a x b x c
    res = np.dot(a, b)
    res = np.dot(res, c)
    return res


def Dtri_matT(a, b):
    # triple matrix multiplication a x D x a'  D is diagonal matrix in row vector form
    res = np.multiply(a, b)
    res = np.dot(res, a.T)
    return res


def Dtri_mat(a, b, c):
    # triple matrix multiplication a x D x c  D is diagonal matrix in row vector form
    res = np.multiply(a, b)
    res = np.dot(res, c)
    return res

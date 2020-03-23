import numpy as np
import logging
import sys


def random_pair(num_snp, out_file='random_pair', num_pair=100000, num_each_pair=5000):
    """
    Produce random SNP pairs for additive by additive epistasis and dominance by dominance epistasis
    :param num_snp: total number of SNPs
    :param out_file: output file. default is 'random_pair'
    :param num_pair: total SNP pairs to produce. default is 10000
    :param num_each_pair: the number of SNP pairs for each sampling. default is 5000
    :return: the random SNP pairs
    """
    if num_pair > num_snp*(num_snp-1)/2:
        logging.error('num_pair must be not greater than: ' + str(num_snp*(num_snp-1)/2))
        sys.exit()
    if num_pair < num_each_pair:
        logging.error('num_pair must be greater than num_each_pair')
        sys.exit()
    res_lst = set()
    while True:
        arr = np.random.randint(num_snp, size=(num_each_pair, 2))
        arr = arr[arr[:, 0] < arr[:, 1], :]
        arr = np.array(arr, dtype=str)
        lst = map(lambda x, y: x + ' ' + y, arr[:, 0], arr[:, 1])
        res_lst.update(set(lst))
        if len(res_lst) >= num_pair:
            break
    res_lst = list(res_lst)
    res_lst = res_lst[:num_pair]
    np.savetxt(out_file, np.array(res_lst), fmt="%s", header='snp_0 snp_1', comments='')
    res_lst = np.loadtxt(out_file, dtype=np.int, skiprows=1)
    return res_lst


def random_pairAD(num_snp, out_file='random_pair', num_pair=100000, num_each_pair=5000):
    """
    Produce random SNP pairs for additive by dominance epistasis
    :param num_snp: total number of SNPs
    :param out_file: output file. default is 'random_pair'
    :param num_pair: total SNP pairs to produce. default is 10000
    :param num_each_pair: the number of SNP pairs for each sampling. default is 5000
    :return: the random SNP pairs
    """
    if num_pair > num_snp*(num_snp-1):
        logging.error('num_pair must be not greater than: ' + str(num_snp*(num_snp-1)))
        sys.exit()
    if num_pair < num_each_pair:
        logging.error('num_pair must be greater than num_each_pair')
        sys.exit()
    res_lst = set()
    while True:
        arr = np.random.randint(num_snp, size=(num_each_pair, 2))
        arr = arr[arr[:, 0] != arr[:, 1], :]
        arr = np.array(arr, dtype=str)
        lst = map(lambda x, y: x + ' ' + y, arr[:, 0], arr[:, 1])
        res_lst.update(set(lst))
        if len(res_lst) >= num_pair:
            break
    res_lst = list(res_lst)
    res_lst = res_lst[:num_pair]
    np.savetxt(out_file, np.array(res_lst), fmt="%s", header='snp_0 snp_1', comments='')
    res_lst = np.loadtxt(out_file, dtype=np.int, skiprows=1)
    return res_lst


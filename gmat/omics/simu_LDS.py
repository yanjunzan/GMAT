import numpy as np
import pandas as pd
import sys
import multiprocessing
import os


def simu_LDS(total_snp, depth, rep):
    np.random.seed(rep*100)
    simu_one = pd.Series(np.random.randint(total_snp, size=int(total_snp*depth)))
    simu_count = simu_one.value_counts()
    simu_count = pd.Series(simu_count, index=range(total_snp))
    simu_count = simu_count.fillna(0)
    return np.array(simu_count).reshape(-1, 1)


def simu(total_snp, depth, num_id, num_processes=20):
    """
    模拟低深度测序
    :param total_snp:
    :param depth:
    :param num_id:
    :param num_processes:
    :return:
    """
    results = []
    pool = multiprocessing.Pool(processes=num_processes)
    for rep in range(num_id):
        results.append(pool.apply_async(simu_LDS, args=(total_snp, depth, rep,)))
    pool.close()
    pool.join()
    simu_vec = []
    for res in results:
        simu_vec.append(res.get())
    simu_df = np.concatenate(simu_vec, axis=1)
    return simu_df


if __name__ == '__main__':
    total_snp = 67333049
    depth = 1
    num_id = 100
    dp_cut = 5
    num_processes = 20
    res = simu(total_snp, depth, num_id, num_processes=num_processes)

'''
total_snp = 6733304
depth = 1
num_id = 1000
num_processes = 50
rep = 0
res = simu(total_snp, depth, num_id, num_processes=num_processes)
sum = res >=3
x = np.sum(sum,axis=0)
y = np.sum(sum,axis=1)
'''



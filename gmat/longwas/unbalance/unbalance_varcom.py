import time
from .unbalance_predata import unbalance_predata
from .unbalance_emai import unbalance_emai


def unbalance_varcom(data_file, id, tpoint, trait, gmat_inv_file, tfix=None, fix=None, forder=3, aorder=3, porder=3,
             na_method='omit', fix_const=False, ran_const=True, init=None, max_iter=200, cc_par=1.0e-8, cc_gra=1.0e6,
                     cc_ll=1.0e6, em_weight_step=0.01, prefix_outfile='glta_unbalance_varcom'):
    print("#" * 100)
    print("{:#^100}".format("Prepare the data for unbalanced longitudinal variances estimation"))
    print("#" * 100)
    clock_t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    y, xmat, zmat, rmat_inv_lst, code_id_dct, id_order, fix_pos_vec = unbalance_predata(data_file, id, tpoint, trait,
                gmat_inv_file, tfix=tfix, fix=fix, forder=forder, aorder=aorder, porder=porder, na_method=na_method)
    clock_t1 = time.perf_counter()
    cpu_t1 = time.process_time()
    print("&&&Running time: Clock time, {:.5f} sec; CPU time, {:.5f} sec.".format(clock_t1 - clock_t0, cpu_t1 - cpu_t0))
    
    print("#" * 100)
    print("{:#^100}".format("variances estimation for unbalanced longitudinal data"))
    print("#" * 100)
    clock_t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    res_df = unbalance_emai(y, xmat, zmat, rmat_inv_lst, fix_const=fix_const, ran_const=ran_const, init=init,
                 max_iter=max_iter, cc_par=cc_par, cc_gra=cc_gra, cc_ll=cc_ll, em_weight_step=em_weight_step)
    res_df['effect_ind'][0] = fix_pos_vec
    clock_t1 = time.perf_counter()
    cpu_t1 = time.process_time()
    print("&&&Running time: Clock time, {:.5f} sec; CPU time, {:.5f} sec.".format(clock_t1 - clock_t0, cpu_t1 - cpu_t0))
    
    var_file = prefix_outfile + '.var'
    res_df['variances'].to_csv(var_file, sep=' ', index=False)
    return res_df

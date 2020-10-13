
### Estimate the variances
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat
from gmat.longwas.balance import balance_varcom
import logging

logging.basicConfig(level=logging.INFO)

bed_file = '../data/mouse_long/plink'
agmat(bed_file, inv=True, small_val=0.001, out_fmt='id_id_val')

data_file = '../data/mouse_long/phe.balance.txt'
id = 'ID'
tpoint = np.array(range(16)) + 1.0
trait = range(2, 18)
kin_file = '../data/mouse_long/plink.agrm2'
prefix_outfile = '../data/mouse_long/balance_varcom'
res_var = balance_varcom(data_file, id, tpoint, trait, kin_file, prefix_outfile=prefix_outfile)
print(res_var)

### longitudinal GWAS by fixed method

import numpy as np
import pandas as pd
from gmat.gmatrix import agmat
from gmat.longwas.balance import balance_varcom
from gmat.longwas.balance import balance_longwas_fixed
from gmat.longwas.balance import balance_longwas_fixed_permutation
import logging

logging.basicConfig(level=logging.INFO)

logging.basicConfig(level=logging.INFO)

bed_file = '../data/mouse_long/plink'
# agmat(bed_file, inv=True, small_val=0.001, out_fmt='id_id_val')

data_file = '../data/mouse_long/phe.balance.txt'
id = 'ID'
tpoint = np.array(range(16)) + 1.0
trait = range(2, 18)
kin_file = '../data/mouse_long/plink.agrm2'
prefix_outfile = '../data/mouse_long/balance_varcom'
# res_var = balance_varcom(data_file, id, tpoint, trait, kin_file, prefix_outfile=prefix_outfile)
var_com = pd.read_csv('../data/mouse_long/balance_varcom.var', header=0, sep='\s+')
prefix_outfile = '../data/mouse_long/balance_longwas_fixed'
res_fixed = balance_longwas_fixed(data_file, id, tpoint, trait, kin_file, bed_file, var_com, maxiter=0, snp_lst=None,
                                 prefix_outfile=prefix_outfile)
res_fixed_perm = balance_longwas_fixed_permutation(data_file, id, tpoint, trait, kin_file, bed_file, var_com, maxiter=0, snp_lst=None,
                                 prefix_outfile=prefix_outfile)


### longitudinal GWAS by trans method

import numpy as np
import pandas as pd
from gmat.gmatrix import agmat
from gmat.longwas.balance import balance_varcom
from gmat.longwas.balance import balance_longwas_trans
from gmat.longwas.balance import balance_longwas_trans_permutation
import logging

logging.basicConfig(level=logging.INFO)

logging.basicConfig(level=logging.INFO)

bed_file = '../data/mouse_long/plink'
# agmat(bed_file, inv=True, small_val=0.001, out_fmt='id_id_val')

data_file = '../data/mouse_long/phe.balance.txt'
id = 'ID'
tpoint = np.array(range(16)) + 1.0
trait = range(2, 18)
kin_file = '../data/mouse_long/plink.agrm2'
prefix_outfile = '../data/mouse_long/balance_varcom'
# res_var = balance_varcom(data_file, id, tpoint, trait, kin_file, prefix_outfile=prefix_outfile)
var_com = pd.read_csv('../data/mouse_long/balance_varcom.var', header=0, sep='\s+')
prefix_outfile = '../data/mouse_long/balance_longwas_trans'
res_trans = balance_longwas_trans(data_file, id, tpoint, trait, kin_file, bed_file, var_com, snp_lst=None,
                                 prefix_outfile=prefix_outfile)
res_trans_perm = balance_longwas_trans_permutation(data_file, id, tpoint, trait, kin_file, bed_file, var_com, snp_lst=None,
                                 prefix_outfile=prefix_outfile)


### balance_longwas_trans + balance_longwas_fixed
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat
from gmat.longwas.balance import balance_varcom
from gmat.longwas.balance import balance_longwas_fixed, balance_longwas_trans
import logging

logging.basicConfig(level=logging.INFO)

logging.basicConfig(level=logging.INFO)

bed_file = '../data/mouse_long/plink'
agmat(bed_file, inv=True, small_val=0.001, out_fmt='id_id_val')

data_file = '../data/mouse_long/phe.balance.txt'
id = 'ID'
tpoint = np.array(range(16)) + 1.0
trait = range(2, 18)
kin_file = '../data/mouse_long/plink.agrm2'
prefix_outfile = '../data/mouse_long/balance_varcom'
res_var = balance_varcom(data_file, id, tpoint, trait, kin_file, prefix_outfile=prefix_outfile)
var_com = pd.read_csv('../data/mouse_long/balance_varcom.var', header=0, sep='\s+')
prefix_outfile = '../data/mouse_long/balance_longwas_trans'
res_trans = balance_longwas_trans(data_file, id, tpoint, trait, kin_file, bed_file, var_com,
                                  prefix_outfile=prefix_outfile)

res_trans_sel = res_trans[res_trans['p_val'] < 0.01]
snp_lst = np.array(res_trans_sel['order'])
prefix_outfile = '../data/mouse_long/balance_longwas_fixed_sel'
res_fixed_sel = balance_longwas_fixed(data_file, id, tpoint, trait, kin_file, bed_file, var_com, maxiter=0, snp_lst=snp_lst,
                                 prefix_outfile=prefix_outfile)

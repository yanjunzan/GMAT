
### gma.cal_kin
from gma import cal_kin
bed_file = './tests/plink'
kin_lst = cal_kin(bed_file)


### gma.balance_varcom
import numpy as np
import pandas as pd
from gma import cal_kin
from gma import balance_varcom
bed_file = './tests/plink'
kin_lst = cal_kin(bed_file)
data_file = './tests/phe.balance.txt'
id = 'ID'
tpoint = np.array(range(16)) + 1.0
trait = range(2, 18)
kin_file = './tests/plink.grm'
prefix_outfile = './tests/gma_balance_varcom'
res_var = balance_varcom(data_file, id, tpoint, trait, kin_file, prefix_outfile=prefix_outfile)
print res_var
# If the variances do not converge, we can use the previous variances as initial values.
init = np.array(res_var['variances']['var_val'])
# or
# var_com = pd.read_csv('./tests/gma_balance_varcom.var', header=0, sep='\s+')
# init = np.array(var_com['var_val'])
res_var = balance_varcom(data_file, id, tpoint, trait, kin_file, init=init, prefix_outfile=prefix_outfile)


### gma.balance_longwas_fixreg
import numpy as np
import pandas as pd
from gma import cal_kin
from gma import balance_varcom
from gma import balance_longwas_fixreg
bed_file = './tests/plink'
kin_lst = cal_kin(bed_file)
data_file = './tests/phe.balance.txt'
id = 'ID'
tpoint = np.array(range(16)) + 1.0
trait = range(2, 18)
kin_file = './tests/plink.grm'
prefix_outfile = './tests/gma_balance_varcom'
res_var = balance_varcom(data_file, id, tpoint, trait, kin_file, prefix_outfile=prefix_outfile)
var_com = res_var['variances']
# or
# var_com = pd.read_csv('./tests/gma_balance_varcom.var', header=0, sep='\s+')
snp_lst = range(1, 10)
# snp_lst = [2, 3, 5, 10]
prefix_outfile = './tests/gma_balance_longwas_fixreg'
res_lst = balance_longwas_fixreg(data_file, id, tpoint, trait, kin_file, bed_file, var_com, snp_lst=snp_lst,
                                 prefix_outfile=prefix_outfile)

### gma.balance_longwas_lt
import numpy as np
import pandas as pd
from gma import cal_kin
from gma import balance_varcom
from gma import balance_longwas_lt
bed_file = './tests/plink'
kin_lst = cal_kin(bed_file)
data_file = './tests/phe.balance.txt'
id = 'ID'
tpoint = np.array(range(16)) + 1.0
trait = range(2, 18)
kin_file = './tests/plink.grm'
prefix_outfile = './tests/gma_balance_varcom'
res_var = balance_varcom(data_file, id, tpoint, trait, kin_file, prefix_outfile=prefix_outfile)
var_com = res_var['variances']
# or
# var_com = pd.read_csv('./tests/gma_balance_varcom.var', header=0, sep='\s+')
snp_lst = range(1, 10)
# snp_lst = [2, 3, 5, 10]
prefix_outfile = './tests/gma_balance_longwas_lt'
res_lst = balance_longwas_lt(data_file, id, tpoint, trait, kin_file, bed_file, var_com, snp_lst=snp_lst,
                                 prefix_outfile=prefix_outfile)


### gma.balance_longwas_lt + gma.balance_longwas_fixreg
import numpy as np
import pandas as pd
from gma import cal_kin
from gma import balance_varcom
from gma import balance_longwas_lt
bed_file = './tests/plink'
kin_lst = cal_kin(bed_file)
data_file = './tests/phe.balance.txt'
id = 'ID'
tpoint = np.array(range(16)) + 1.0
trait = range(2, 18)
kin_file = './tests/plink.grm'
prefix_outfile = './tests/gma_balance_varcom'
res_var = balance_varcom(data_file, id, tpoint, trait, kin_file, prefix_outfile=prefix_outfile)
var_com = res_var['variances']
# or
# var_com = pd.read_csv('./tests/gma_balance_varcom.var', header=0, sep='\s+')
prefix_outfile = './tests/gma_balance_longwas_lt'
res_lst = balance_longwas_lt(data_file, id, tpoint, trait, kin_file, bed_file, var_com, snp_lst=None,
                                 prefix_outfile=prefix_outfile)
res_lst_sel = res_lst[res_lst['p_val'] < 0.01]
snp_lst = np.array(res_lst_sel['order'])
prefix_outfile = './tests/gma_balance_longwas_fixreg_sel'
res_lst_sel = balance_longwas_fixreg(data_file, id, tpoint, trait, kin_file, bed_file, var_com, snp_lst=snp_lst,
                                 prefix_outfile=prefix_outfile)



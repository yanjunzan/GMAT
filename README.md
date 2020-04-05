
# Contents
* [1 Contact](#1-Contact)
* [2 Install](#2-Install)
  * [2.1 Dependencies](#21-Dependencies)
  * [2.2 Quick install](#22-Quick-install)
  * [2.3 Detailed Package Install Instructions](#23-Detailed-Package-Install-Instructions)
* [3 REMMAX function](#3-REMMAX-function)
  * [3.1 Format of the input file](#31-Format-of-the-input-file)
  * [3.2 Exhaustive additive by addtive epistatis](#32-Exhaustive-additive-by-addtive-epistatis)
    * [3.2.1 Include additive and additive by additive genomic relationship matrix](#321-Include-additive-and-additive-by-additive-genomic-relationship-matrix)
    * [3.2.2 Include additive, dominance and additive by additive genomic relationship matrix](#322-Include-additive-dominance-and-additive-by-additive-genomic-relationship-matrix)
    * [3.2.3 Include additive, dominance and three kinds of epistatic genomic relationship matrix](#323-Include-additive-dominance-and-three-kinds-of-epistatic-genomic-relationship-matrix)
  * [3.3 Exhaustive additive by dominance epistatis](#33-Exhaustive-additive-by-dominance-epistatis)
  * [3.4 Exhaustive dominance by dominance epistatis](#34-Exhaustive-dominance-by-dominance-epistatis)

# 1 Contact
Chao Ning  
ningchao(at)sdau(dot)edu(dot)cn  
ningchao91(at)gmail(dot)com  

# 2 Install
We will keep GMAT updated. Please uninstall older version to obtain the latest functions. The easiest uninstall way:  
\> pip uninstall gmat

## 2.1 Dependencies
* numpy>=1.16.0  
* pandas>=0.19.0  
* scipy>=1.1.1  
* cffi>=1.12.0  
* pandas_plink>=2.0.0  
* tqdm>=4.43.0  

We recommend using a Python distribution such as [Anaconda](https://www.anaconda.com/distribution/) (Python 3.7 version). This distribution can be used on Linux and Windows. It is the easiest way to get all the required package dependencies. 

## 2.2 Quick install
\> pip install gmat  

## 2.3 Detailed Package Install Instructions
(1) Install the dependent packages  
(2) Go to the directory of GMAT and type  
\> python setup.py install  

# 3 REMMAX function  
Rapid Epistatic Mixed Model Association Studies

***Cite***:  
* Dan Wang, Hui Tang, Jian-Feng Liu, Shizhong Xu, Qin Zhang and Chao Ning. Rapid Epistatic Mixed Model Association Studies by Controlling Multiple Polygenic Effects. *BioRxiv*, 2020. doi: https://doi.org/10.1101/2020.03.05.976498  
* Chao Ning, Dan Wang, Huimin Kang, Raphael Mrode, Lei Zhou, Shizhong Xu, Jian-Feng Liu. A rapid epistatic mixed-model association analysis by linear retransformations of genomic estimated values. *Bioinformatics*, 2018, 34(11): 1817-1825.  

## 3.1 Format of the input file
* Plink binary file including \*.bed, \*.bim and \*.fam.  
Missing genotypes are recommended to impute with Beagle or other softwares, although they will be imputed according the frequency of occurrence.   

* phenotypic file:  
(1) Delimited by blanks or tabs;  
(2) All individuals in the plink file must have phenotypic values. If no, please remove these individuals from the plink binary file;  
(3) The fisrt column is the family id and the second column is the individual id. The first two columns are the same to plink fam file, but order can be different;  
(4) The last column is the phenotypic values. **Miss values are not allowed**;  
(5) The covariates (including population means) are put before the phenotypic column. A column of 1’s must be contained.  
(6) Repeated mesures are allowed for individuals.
An example phenotypic file with four covariates (population mean, sex, age, treatmeant or untreatmeant) is as follows:  
12659	14462	1	0	126	0	0.58  
12659	14463	1	0	91	1	0.39  
12659	14464	1	1	126	0	0.37  
12659	14465	1	0	91	1	0.9  
12659	14466	1	0	91	1	0.84  
12659	14467	1	0	91	1	0.61  
12659	14468	1	1	91	1	0.84  
An example phenotypic file with repeated mesures is as follows:  
0 01_01 1 -2.25383070574996  
0 01_02 1 -1.88774565927855  
0 01_03 1 2.4150679267528  
0 01_03 1 -0.320697695608065  
0 01_04 1 2.41743663901475  
0 01_06 1 -0.634513668596019  
0 01_06 1 -1.4489729404784  
0 01_07 1 1.92328500921367  
0 01_07 1 1.54547583777757  
  
## 3.2 Exhaustive additive by addtive epistatis  
Data: Mouse data in directory of GMAT/examples/data/mouse    

### 3.2.1 Include additive and additive by additive genomic relationship matrix
#### (1)  Exact test (for small data)


```python
import logging
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAA import remma_epiAA
from gmat.remma import annotation_snp_pos


logging.basicConfig(level=logging.INFO)
pheno_file = 'pheno'  # phenotypic file
bed_file = 'plink'  # the prefix for the plink binary file

# Step 1: Calculate the genomic relationship matrix
agmat(bed_file) 
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix

# Step 2: Prepare the phenotypic vector (y), designed matrix for fixed effects (xmat) and designed matrix for random effects (zmat)
y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)

# Step 3: Estimate the variances
gmat_lst = [ag, ag*ag]  # ag*ag is the additive by additive genomic relationship matrix
var_a_axa = wemai_multi_gmat(y, xmat, zmat, gmat_lst)
print(var_a_axa)  # a list： [0] addtive variance; [1] additive by additive variance; [2] residual variance

# Step 4: Test
remma_epiAA(y, xmat, zmat, gmat_lst, var_com=var_a_axa, bed_file=bed_file, p_cut=0.0001, out_file='epiAA_a_axa')

# Step 5: Select top SNPs and add the SNP position
res_file = 'epiAA_a_axa'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

#### (2) Parallel exact test  (for small data)
Analysis can be subdivided with remma_epiAA_parallel and run parallelly on different machines.


```python
import logging
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAA import remma_epiAA_parallel
from gmat.remma import annotation_snp_pos

logging.basicConfig(level=logging.INFO)
pheno_file = 'pheno'  # phenotypic file
bed_file = 'plink'  # the prefix for the plink binary file

# Write codes of step 1-4 in separate scripts and run separately
# Step 1-3 is same to the above

# Step 4: parallel test.
remma_epiAA_parallel(y, xmat, zmat, gmat_lst, var_com=var_a_axa, bed_file=bed_file, parallel=[3,1], 
                         p_cut=0.0001, out_file='epiAA_parallel_a_axa')
remma_epiAA_parallel(y, xmat, zmat, gmat_lst, var_com=var_a_axa, bed_file=bed_file, parallel=[3,2], 
                         p_cut=0.0001, out_file='epiAA_parallel_a_axa')
remma_epiAA_parallel(y, xmat, zmat, gmat_lst, var_com=var_a_axa, bed_file=bed_file, parallel=[3,3], 
                         p_cut=0.0001, out_file='epiAA_parallel_a_axa')

# Step 5: Merge files 'epiAA_parallel_a_axa.1', 'epiAA_parallel_a_axa.2' and 'epiAA_parallel_a_axa.3' with the following codes.
prefix = 'epiAA_parallel_a_axa'
parallel_num = 3  # the number of parallels
with open(prefix + ".merge", 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)

# Step 6: Select top SNPs and add the SNP position
res_file = 'epiAA_parallel_a_axa.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0

```

#### (3) approximate test (recommended for big data)


```python
import logging
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma import random_pair
from gmat.remma.remma_epiAA import remma_epiAA_pair, remma_epiAA_eff
from gmat.remma import annotation_snp_pos

logging.basicConfig(level=logging.INFO)
pheno_file = 'pheno'  # phenotypic file
bed_file = 'plink'  # the prefix for the plink binary file

# Step 1-3 is same to exact test

# Step 4: Randomly select 100,000 SNP pairs
snp_df = pd.read_csv(bed_file + '.bim', header=None, sep='\s+')
num_snp = snp_df.shape[0]  # the number of snp
random_pair(num_snp, out_file='random_pair', num_pair=100000, num_each_pair=5000)

# step 5: Test these 100,000 SNP pairs
# note: set p_cut=1 to save all the results
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_a_axa, bed_file=bed_file, snp_pair_file="random_pair", 
                     max_test_pair=50000, p_cut=1, out_file='epiAA_pair_random_a_axa')

# step 6: Calculate the median of variances for estimated epistatic SNP effects
res_df = pd.read_csv('epiAA_pair_random_a_axa', header=0, sep='\s+')
print(np.median(res_df['p']))  # P value close to 0.5. It means type I error controlled well
var_median = np.median(res_df['var'])  # median of variances for estimated epistatic SNP effects

# step 7: Screen the effects and select top SNP pairs based on approximate test. 
# Use the above median of variances as the approximate values (var_app = var_median)
remma_epiAA_eff(y, xmat, zmat, gmat_lst, var_com=var_a_axa, bed_file=bed_file, var_app=var_median, 
                      p_cut=1e-05, out_file='epiAA_eff_a_axa')

# Step 8: Calculate exact p values for top SNP pairs
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_a_axa, bed_file=bed_file, snp_pair_file="epiAA_eff_a_axa", 
                     max_test_pair=50000, p_cut=1, out_file='epiAA_pair_res_a_axa')

# Step 9: Select top SNPs and add the SNP position
res_file = 'epiAA_pair_res_a_axa'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0

```

#### (4) Parallel approximate test (recommended for big data)
Analysis can be subdivided with remma_epiAA_eff_parallel and run parallelly on different machines.


```python
import logging
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma import random_pair
from gmat.remma.remma_epiAA import remma_epiAA_pair, remma_epiAA_eff
from gmat.remma import annotation_snp_pos

logging.basicConfig(level=logging.INFO)
pheno_file = 'pheno'  # phenotypic file
bed_file = 'plink'  # the prefix for the plink binary file

# Step 1-3 is same to exact test

# Step 4: Randomly select 100,000 SNP pairs
snp_df = pd.read_csv(bed_file + '.bim', header=None, sep='\s+')
num_snp = snp_df.shape[0]  # the number of snp
random_pair(num_snp, out_file='random_pair', num_pair=100000, num_each_pair=5000)

# step 5: Test these 100,000 SNP pairs
# note: set p_cut=1 to save all the results
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_a_axa, bed_file=bed_file, snp_pair_file="random_pair", 
                     max_test_pair=50000, p_cut=1, out_file='epiAA_pair_random_a_axa')

# step 6: Calculate the median of variances for estimated epistatic SNP effects
res_df = pd.read_csv('epiAA_pair_random_a_axa', header=0, sep='\s+')
print(np.median(res_df['p']))  # P value close to 0.5. It means type I error controlled well
var_median = np.median(res_df['var'])  # median of variances for estimated epistatic SNP effects

# step 7: Screen the effects and select top SNP pairs based on approximate test. 
# Use the above median of variances as the approximate values (var_app = var_median)
remma_epiAA_eff(y, xmat, zmat, gmat_lst, var_com=var_a_axa, bed_file=bed_file, var_app=var_median, 
                      p_cut=1e-05, out_file='epiAA_eff_a_axa')

# Step 8: Calculate exact p values for top SNP pairs
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_a_axa, bed_file=bed_file, snp_pair_file="epiAA_eff_a_axa", 
                     max_test_pair=50000, p_cut=1, out_file='epiAA_pair_res_a_axa')

# Step 9: Select top SNPs and add the SNP position
res_file = 'epiAA_pair_res_a_axa'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0


```

### 3.2.2 Include additive, dominance and additive by additive genomic relationship matrix
#### (1) Exact test (for small data)


```python
import logging
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAA import remma_epiAA, remma_epiAA_parallel
from gmat.remma import annotation_snp_pos
logging.basicConfig(level=logging.INFO)

pheno_file = 'pheno'
bed_file = 'plink'

# Step 1: Calculate the genomic relationship matrix
agmat_lst = agmat(bed_file, inv=False) # additive genomic relationship matrix
dgmat_lst = dgmat_as(bed_file, inv=False) # dominace genomic relationship matrix

# Step 2: Prepare the phenotypic vector (y), designed matrix for fixed effects (xmat) and designed matrix for random effects (zmat)
y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)

# Step 3: Estimate the variances
gmat_lst = [agmat_lst[0], dgmat_lst[0], agmat_lst[0]*agmat_lst[0]]  # agmat_lst[0]*agmat_lst[0] is the additive by additive genomic relationship matrix
var_com_a_d_axa = wemai_multi_gmat(y, xmat, zmat, gmat_lst)
print(var_com_a_d_axa)  # a list： [0] addtive variance; [1] dominace variance [2] additive by additive variance; [3] residual variance

# Step 4: Test
remma_epiAA(y, xmat, zmat, gmat_lst, var_com=var_com_a_d_axa, bed_file=bed_file, p_cut=0.0001, out_file='remma_epiAA_a_d_axa')

# Step 5: Select top SNPs and add the SNP position
res_file = 'remma_epiAA_a_d_axa'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

#### (2) Parallel exact test (for small data)
Analysis can be subdivided with remma_epiAA_parallel and run parallelly on different machines.


```python
# Write codes of step 1-4 in separate scripts and run separately
# Step 1-3 is same to the above

# Step 4: parallel test. Write the codes in separate scripts and run separately.
from gmat.remma.remma_epiAA import remma_epiAA_parallel
remma_epiAA_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_a_d_axa, bed_file=bed_file, parallel=[3,1], 
                         p_cut=0.0001, out_file='remma_epiAA_parallel_a_d_axa')
remma_epiAA_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_a_d_axa, bed_file=bed_file, parallel=[3,2], 
                         p_cut=0.0001, out_file='remma_epiAA_parallel_a_d_axa')
remma_epiAA_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_a_d_axa, bed_file=bed_file, parallel=[3,3], 
                         p_cut=0.0001, out_file='remma_epiAA_parallel_a_d_axa')

# Step 5: Merge files 'remma_epiAA_parallel_a_d_axa.1', 'remma_epiAA_parallel_a_d_axa.2' and 'remma_epiAA_parallel_a_d_axa.3' with the following codes.
prefix = 'remma_epiAA_parallel_a_d_axa'
parallel_num = 3  # the number of parallels
with open(prefix + ".merge", 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)

# Step 6: Select top SNPs and add the SNP position
res_file = 'remma_epiAA_parallel_a_d_axa.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0

```

#### (3) approximate test (recommended for big data)


```python
import logging
import numpy as np
import pandas as pd
from scipy.stats import chi2
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma import random_pair
from gmat.remma.remma_epiAA import remma_epiAA_pair, remma_epiAA_eff, remma_epiAA_eff_parallel
from gmat.remma import annotation_snp_pos
logging.basicConfig(level=logging.INFO)

pheno_file = 'pheno'  # phenotypic file
bed_file = 'plink'  # the prefix for the plink binary file

# Step 1-3 is same to exact test

# Step 4: Randomly select 100,000 SNP pairs
snp_df = pd.read_csv(bed_file + '.bim', header=None, sep='\s+')
num_snp = snp_df.shape[0]  # the number of snp
random_pair(num_snp, out_file='random_pair', num_pair=100000, num_each_pair=5000)

# step 5: Test these 100,000 SNP pairs
# note: set p_cut=1 to save all the results
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_com_a_d_axa, bed_file=bed_file, snp_pair_file="random_pair", 
                     max_test_pair=50000, p_cut=1, out_file='remma_epiAA_pair_random_a_d_axa')

# step 6: Calculate the median of variances for estimated epistatic SNP effects
res_df = pd.read_csv('remma_epiAA_pair_random_a_d_axa', header=0, sep='\s+')
print(np.median(res_df['p']))  # P value close to 0.5. It means type I error controlled well
var_median = np.median(res_df['var'])  # median of variances for estimated epistatic SNP effects

# step 7: Screen the effects and select top SNP pairs based on approximate test. 
# Use the above median of variances as the approximate values (var_app = var_median)
remma_epiAA_eff(y, xmat, zmat, gmat_lst, var_com=var_com_a_d_axa, bed_file=bed_file, var_app=var_median, 
                      p_cut=1e-05, out_file='remma_epiAA_eff_a_d_axa')

# Step 8: Calculate exact p values for top SNP pairs
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_com_a_d_axa, bed_file=bed_file, snp_pair_file="remma_epiAA_eff_a_d_axa", 
                     max_test_pair=50000, p_cut=1, out_file='remma_epiAA_pair_res_a_d_axa')

# Step 9: Select top SNPs and add the SNP position
res_file = 'remma_epiAA_pair_res_a_d_axa'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

#### (4) Parallel approximate test (recommended for big data)
Analysis can be subdivided with remma_epiAA_eff_parallel and run parallelly on different machines.


```python
# Write codes of step 1-8 in separate scripts and run separately
# Step 1-6 is same to the above

# Step 7: parallel test. Write the codes in separate scripts and run separately.
from gmat.remma.remma_epiAA import remma_epiAA_eff_parallel
remma_epiAA_eff_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_a_d_axa, bed_file=bed_file, parallel=[3,1], 
                               var_app=var_median, p_cut=1.0e-5, out_file='remma_epiAA_eff_parallel_a_d_axa')
remma_epiAA_eff_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_a_d_axa, bed_file=bed_file, parallel=[3,2], 
                               var_app=var_median, p_cut=1.0e-5, out_file='remma_epiAA_eff_parallel_a_d_axa')
remma_epiAA_eff_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_a_d_axa, bed_file=bed_file, parallel=[3,3], 
                               var_app=var_median, p_cut=1.0e-5, out_file='remma_epiAA_eff_parallel_a_d_axa')

# Step 8: Calculate exact p values for top SNP pairs
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_com_a_d_axa, bed_file=bed_file, snp_pair_file="remma_epiAA_eff_parallel_a_d_axa.1", 
                     max_test_pair=50000, p_cut=1, out_file='remma_epiAA_pair_parallel_a_d_axa.1')
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_com_a_d_axa, bed_file=bed_file, snp_pair_file="remma_epiAA_eff_parallel_a_d_axa.2", 
                     max_test_pair=50000, p_cut=1, out_file='remma_epiAA_pair_parallel_a_d_axa.2')
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_com_a_d_axa, bed_file=bed_file, snp_pair_file="remma_epiAA_eff_parallel_a_d_axa.3", 
                     max_test_pair=50000, p_cut=1, out_file='remma_epiAA_pair_parallel_a_d_axa.3')

# Step 9: Merge files 'epiAA_pair_res_a_d_axa.1', 'epiAA_pair_res_a_d_axa.2' and 'epiAA_pair_res_a_d_axa.3' 
# with the following codes.
prefix = 'remma_epiAA_pair_parallel_a_d_axa'
parallel_num = 3  # the number of parallels
with open(prefix + ".merge", 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1:
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)

# Step 10: Select top SNPs and add the SNP position
res_file = 'remma_epiAA_pair_parallel_a_d_axa.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
                   
```

### 3.2.3 Include additive, dominance and three kinds of epistatic genomic relationship matrix
additive, dominance, additive by additive, additive by dominance and dominance by dominance genomic relationship matrix   
#### (1) Exact test (for small data)


```python
import logging
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAA import remma_epiAA, remma_epiAA_parallel
from gmat.remma import annotation_snp_pos
logging.basicConfig(level=logging.INFO)

pheno_file = 'pheno'
bed_file = 'plink'

# Step 1: Calculate the genomic relationship matrix
agmat_lst = agmat(bed_file, inv=False) # additive genomic relationship matrix
dgmat_lst = dgmat_as(bed_file, inv=False) # dominace genomic relationship matrix

# Step 2: Prepare the phenotypic vector (y), designed matrix for fixed effects (xmat) and designed matrix for random effects (zmat)
y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)

# Step 3: Estimate the variances
# agmat_lst[0]*agmat_lst[0] is the additive by additive genomic relationship matrix
# agmat_lst[0]*dgmat_lst[0] is the additive by dominance genomic relationship matrix
# dgmat_lst[0]*dgmat_lst[0] is the dominance by dominance genomic relationship matrix
gmat_lst = [agmat_lst[0], dgmat_lst[0], agmat_lst[0]*agmat_lst[0], agmat_lst[0]*dgmat_lst[0], dgmat_lst[0]*dgmat_lst[0]]  

var_com_all = wemai_multi_gmat(y, xmat, zmat, gmat_lst)
print(var_com_all)  # a list： [0] addtive variance; [1] dominace variance [2] additive by additive variance; [3] residual variance

# Step 4: Test
remma_epiAA(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, p_cut=0.0001, out_file='remma_epiAA')

# Step 5: Select top SNPs and add the SNP position
res_file = 'remma_epiAA'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5)
```

#### (2) Parallel exact test (for small data)
Analysis can be subdivided with remma_epiAA_parallel and run parallelly on different machines.


```python
# Step 1-3 is same to the above

# Step 4: parallel test. Write the codes in separate scripts and run separately.
from gmat.remma.remma_epiAA import remma_epiAA_parallel
remma_epiAA_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,1], 
                         p_cut=0.0001, out_file='remma_epiAA_parallel')
remma_epiAA_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,2], 
                         p_cut=0.0001, out_file='remma_epiAA_parallel')
remma_epiAA_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,3], 
                         p_cut=0.0001, out_file='remma_epiAA_parallel')

# Step 5: Merge files 'remma_epiAA_parallel.1', 'remma_epiAA_parallel.2' and 'remma_epiAA_parallel.3' with the following codes.
prefix = 'remma_epiAA_parallel'
parallel_num = 3  # the number of parallels
with open(prefix + ".merge", 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)

# Step 6: Select top SNPs and add the SNP position
res_file = 'remma_epiAA_parallel.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5)
```

#### (3) approximate test (recommended for big data)



```python
import logging
import numpy as np
import pandas as pd
from scipy.stats import chi2
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma import random_pair
from gmat.remma.remma_epiAA import remma_epiAA_pair, remma_epiAA_eff, remma_epiAA_eff_parallel
from gmat.remma import annotation_snp_pos
logging.basicConfig(level=logging.INFO)

pheno_file = 'pheno'  # phenotypic file
bed_file = 'plink'  # the prefix for the plink binary file

# Step 1-3 is same to exact test

# Step 4: Randomly select 100,000 SNP pairs
snp_df = pd.read_csv(bed_file + '.bim', header=None, sep='\s+')
num_snp = snp_df.shape[0]  # the number of snp
random_pair(num_snp, out_file='random_pair', num_pair=100000, num_each_pair=5000)

# step 5: Test these 100,000 SNP pairs
# note: set p_cut=1 to save all the results
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="random_pair", 
                     max_test_pair=50000, p_cut=1, out_file='remma_epiAA_pair_random')

# step 6: Calculate the median of variances for estimated epistatic SNP effects
res_df = pd.read_csv('remma_epiAA_pair_random', header=0, sep='\s+')
print(np.median(res_df['p']))  # P value close to 0.5. It means type I error controlled well
var_median = np.median(res_df['var'])  # median of variances for estimated epistatic SNP effects

# step 7: Screen the effects and select top SNP pairs based on approximate test. 
# Use the above median of variances as the approximate values (var_app = var_median)
remma_epiAA_eff(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, var_app=var_median, 
                      p_cut=1e-05, out_file='remma_epiAA_eff')

# Step 8: Calculate exact p values for top SNP pairs
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="remma_epiAA_eff", 
                     max_test_pair=50000, p_cut=1, out_file='remma_epiAA_pair_res')

# Step 9: Select top SNPs and add the SNP position
res_file = 'remma_epiAA_pair_res'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5)

```

#### (4) Parallel approximate test (recommended for big data)
Analysis can be subdivided with remma_epiAA_eff_parallel and run parallelly on different machines.


```python
# Step 1-6 is same to the above

# Step 7: parallel test. Write the codes in separate scripts and run separately.
from gmat.remma.remma_epiAA import remma_epiAA_eff_parallel
remma_epiAA_eff_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,1], 
                               var_app=var_median, p_cut=1.0e-5, out_file='remma_epiAA_eff_parallel')
remma_epiAA_eff_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,2], 
                               var_app=var_median, p_cut=1.0e-5, out_file='remma_epiAA_eff_parallel')
remma_epiAA_eff_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,3], 
                               var_app=var_median, p_cut=1.0e-5, out_file='remma_epiAA_eff_parallel')

# Step 8: Calculate exact p values for top SNP pairs
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="remma_epiAA_eff_parallel.1", 
                     max_test_pair=50000, p_cut=1, out_file='remma_epiAA_pair_res.1')
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="remma_epiAA_eff_parallel.2", 
                     max_test_pair=50000, p_cut=1, out_file='remma_epiAA_pair_res.2')
remma_epiAA_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="remma_epiAA_eff_parallel.3", 
                     max_test_pair=50000, p_cut=1, out_file='remma_epiAA_pair_res.3')

# Step 9: Merge files 'remma_epiAA_pair_res.1', 'remma_epiAA_pair_res.2' and 'remma_epiAA_pair_res.3' 
# with the following codes.
prefix = 'remma_epiAA_pair_res'
parallel_num = 3  # the number of parallels
with open(prefix + ".merge", 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)

# Step 10: Select top SNPs and add the SNP position
res_file = 'remma_epiAA_pair_res.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5)

```

## 3.3 Exhaustive additive by dominance epistatis  
Include additive, dominance, additive by additive, additive by dominance and dominance by dominance genomic relationship matrix  
#### (1) Exact test (for small data)


```python
import logging
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAD import remma_epiAD, remma_epiAD_parallel
from gmat.remma import annotation_snp_pos
logging.basicConfig(level=logging.INFO)

pheno_file = 'pheno'
bed_file = 'plink'

# Step 1: Calculate the genomic relationship matrix
agmat_lst = agmat(bed_file, inv=False) # additive genomic relationship matrix
dgmat_lst = dgmat_as(bed_file, inv=False) # dominace genomic relationship matrix

# Step 2: Prepare the phenotypic vector (y), designed matrix for fixed effects (xmat) and designed matrix for random effects (zmat)
y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)

# Step 3: Estimate the variances
# agmat_lst[0]*agmat_lst[0] is the additive by additive genomic relationship matrix
# agmat_lst[0]*dgmat_lst[0] is the additive by dominance genomic relationship matrix
# dgmat_lst[0]*dgmat_lst[0] is the dominance by dominance genomic relationship matrix
gmat_lst = [agmat_lst[0], dgmat_lst[0], agmat_lst[0]*agmat_lst[0], agmat_lst[0]*dgmat_lst[0], dgmat_lst[0]*dgmat_lst[0]]  

var_com_all = wemai_multi_gmat(y, xmat, zmat, gmat_lst)
print(var_com_all)  
"""
a list： [0] addtive variance; [1] dominace variance [2] additive by additive variance; [3] additive by dominance variance; 
[4] dominance by dominance variance; [5] residual variance
"""

# Step 4: Test
remma_epiAD(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, p_cut=0.0001, out_file='epiAD')

# Step 5: Select top SNPs and add the SNP position
res_file = 'remma_epiAD'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5)

```

#### (2) Parallel exact test (for small data)
Analysis can be subdivided with remma_epiAD_parallel and run parallelly on different machines.


```python
# Step 1-3 is same to the above

# Step 4: parallel test. Write the codes in separate scripts and run separately.
from gmat.remma.remma_epiAD import remma_epiAD_parallel
remma_epiAD_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,1], 
                         p_cut=0.0001, out_file='epiAD_parallel')
remma_epiAD_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,2], 
                         p_cut=0.0001, out_file='epiAD_parallel')
remma_epiAD_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,3], 
                         p_cut=0.0001, out_file='epiAD_parallel')

# Step 5: Merge files 'epiAD_parallel.1', 'epiAD_parallel.2' and 'epiAD_parallel.3' with the following codes.
prefix = 'epiAD_parallel'
parallel_num = 3  # the number of parallels
with open(prefix + ".merge", 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)

# Step 6: Select top SNPs and add the SNP position
res_file = 'epiAD_parallel.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5)
```

#### (3) approximate test (recommended for big data)


```python
import logging
import numpy as np
import pandas as pd
from scipy.stats import chi2
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma import random_pairAD
from gmat.remma.remma_epiAD import remma_epiAD_pair, remma_epiAD_eff, remma_epiAD_eff_parallel
from gmat.remma import annotation_snp_pos
logging.basicConfig(level=logging.INFO)

pheno_file = 'pheno'  # phenotypic file
bed_file = 'plink'  # the prefix for the plink binary file

# Step 1-3 is same to exact test

# Step 4: Randomly select 100,000 SNP pairs
snp_df = pd.read_csv(bed_file + '.bim', header=None, sep='\s+')
num_snp = snp_df.shape[0]  # the number of snp
random_pairAD(num_snp, out_file='random_pairAD', num_pair=100000, num_each_pair=5000)

# step 5: Test these 100,000 SNP pairs
# note: set p_cut=1 to save all the results
remma_epiAD_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="random_pairAD", 
                     max_test_pair=50000, p_cut=1, out_file='epiAD_pair_random')

# step 6: Calculate the median of variances for estimated epistatic SNP effects
res_df = pd.read_csv('epiAD_pair_random', header=0, sep='\s+')
print(np.median(res_df['p']))  # P value close to 0.5. It means type I error controlled well
var_median = np.median(res_df['var'])  # median of variances for estimated epistatic SNP effects

# step 7: Screen the effects and select top SNP pairs based on approximate test. 
# Use the above median of variances as the approximate values (var_app = var_median)
remma_epiAD_eff(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, var_app=var_median, 
                      p_cut=1e-05, out_file='epiAD_eff')

# Step 8: Calculate exact p values for top SNP pairs
remma_epiAD_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="epiAD_eff", 
                     max_test_pair=50000, p_cut=1, out_file='epiAD_pair_res')

# Step 9: Select top SNPs and add the SNP position
res_file = 'epiAD_pair_res'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5)
```

#### (4) Parallel approximate test (recommended for big data)
Analysis can be subdivided with remma_epiAD_eff_parallel and run parallelly on different machines.


```python
# Step 1-6 is same to the above

# Step 7: parallel test. Write the codes in separate scripts and run separately.
from gmat.remma.remma_epiAD import remma_epiAD_eff_parallel
remma_epiAD_eff_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,1], 
                               var_app=var_median, p_cut=1.0e-5, out_file='epiAD_eff_parallel')
remma_epiAD_eff_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,2], 
                               var_app=var_median, p_cut=1.0e-5, out_file='epiAD_eff_parallel')
remma_epiAD_eff_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,3], 
                               var_app=var_median, p_cut=1.0e-5, out_file='epiAD_eff_parallel')

# Step 8: Calculate exact p values for top SNP pairs
remma_epiAD_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="epiAD_eff_parallel.1", 
                     max_test_pair=50000, p_cut=1, out_file='epiAD_pair_res.1')
remma_epiAD_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="epiAD_eff_parallel.2", 
                     max_test_pair=50000, p_cut=1, out_file='epiAD_pair_res.2')
remma_epiAD_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="epiAD_eff_parallel.3", 
                     max_test_pair=50000, p_cut=1, out_file='epiAD_pair_res.3')

# Step 9: Merge files 'epiAD_pair_res.1', 'epiAD_pair_res.2' and 'epiAD_pair_res.3' 
# with the following codes.
prefix = 'epiAD_pair_res'
parallel_num = 3  # the number of parallels
with open(prefix + ".merge", 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)

# Step 10: Select top SNPs and add the SNP position
res_file = 'epiAD_pair_res.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5)
```

## 3.4 Exhaustive dominance by dominance epistatis  
Include additive, dominance, additive by additive, additive by dominance and dominance by dominance genomic relationship matrix  
#### (1) Exact test (for small data)


```python
import logging
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiDD import remma_epiDD, remma_epiDD_parallel
from gmat.remma import annotation_snp_pos
logging.basicConfig(level=logging.INFO)

pheno_file = 'pheno'
bed_file = 'plink'

# Step 1: Calculate the genomic relationship matrix
agmat_lst = agmat(bed_file, inv=False) # additive genomic relationship matrix
dgmat_lst = dgmat_as(bed_file, inv=False) # dominace genomic relationship matrix

# Step 2: Prepare the phenotypic vector (y), designed matrix for fixed effects (xmat) and designed matrix for random effects (zmat)
y, xmat, zmat = design_matrix_wemai_multi_gmat(pheno_file, bed_file)

# Step 3: Estimate the variances
# agmat_lst[0]*agmat_lst[0] is the additive by additive genomic relationship matrix
# agmat_lst[0]*dgmat_lst[0] is the additive by dominance genomic relationship matrix
# dgmat_lst[0]*dgmat_lst[0] is the dominance by dominance genomic relationship matrix
gmat_lst = [agmat_lst[0], dgmat_lst[0], agmat_lst[0]*agmat_lst[0], agmat_lst[0]*dgmat_lst[0], dgmat_lst[0]*dgmat_lst[0]]  

var_com_all = wemai_multi_gmat(y, xmat, zmat, gmat_lst)
print(var_com_all)  
"""
a list： [0] addtive variance; [1] dominace variance [2] additive by additive variance; [3] additive by dominance variance; 
[4] dominance by dominance variance; [5] residual variance
"""

# Step 4: Test
remma_epiDD(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, p_cut=0.0001, out_file='epiDD')

# Step 5: Select top SNPs and add the SNP position
res_file = 'remma_epiDD'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5)
```

#### (2) Parallel exact test (for small data)
Analysis can be subdivided with remma_epiDD_parallel and run parallelly on different machines.


```python
# Step 1-3 is same to the above

# Step 4: parallel test. Write the codes in separate scripts and run separately.
from gmat.remma.remma_epiDD import remma_epiDD_parallel
remma_epiDD_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,1], 
                         p_cut=0.0001, out_file='epiDD_parallel')
remma_epiDD_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,2], 
                         p_cut=0.0001, out_file='epiDD_parallel')
remma_epiDD_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,3], 
                         p_cut=0.0001, out_file='epiDD_parallel')

# Step 5: Merge files 'epiAD_parallel.1', 'epiAD_parallel.2' and 'epiAD_parallel.3' with the following codes.
prefix = 'epiDD_parallel'
parallel_num = 3  # the number of parallels
with open(prefix + ".merge", 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)

# Step 6: Select top SNPs and add the SNP position
res_file = 'epiDD_parallel.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5)
```

#### (3) approximate test (recommended for big data)


```python
import logging
import numpy as np
import pandas as pd
from scipy.stats import chi2
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma import random_pair
from gmat.remma.remma_epiDD import remma_epiDD_pair, remma_epiDD_eff, remma_epiDD_eff_parallel
from gmat.remma import annotation_snp_pos
logging.basicConfig(level=logging.INFO)

pheno_file = 'pheno'  # phenotypic file
bed_file = 'plink'  # the prefix for the plink binary file

# Step 1-3 is same to exact test

# Step 4: Randomly select 100,000 SNP pairs
snp_df = pd.read_csv(bed_file + '.bim', header=None, sep='\s+')
num_snp = snp_df.shape[0]  # the number of snp
random_pair(num_snp, out_file='random_pair', num_pair=100000, num_each_pair=5000)

# step 5: Test these 100,000 SNP pairs
# note: set p_cut=1 to save all the results
remma_epiDD_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="random_pair", 
                     max_test_pair=50000, p_cut=1, out_file='epiDD_pair_random')

# step 6: Calculate the median of variances for estimated epistatic SNP effects
res_df = pd.read_csv('epiDD_pair_random', header=0, sep='\s+')
print(np.median(res_df['p']))  # P value close to 0.5. It means type I error controlled well
var_median = np.median(res_df['var'])  # median of variances for estimated epistatic SNP effects

# step 7: Screen the effects and select top SNP pairs based on approximate test. 
# Use the above median of variances as the approximate values (var_app = var_median)
remma_epiDD_eff(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, var_app=var_median, 
                      p_cut=1e-05, out_file='epiDD_eff')

# Step 8: Calculate exact p values for top SNP pairs
remma_epiDD_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="epiDD_eff", 
                     max_test_pair=50000, p_cut=1, out_file='epiDD_pair_res')

# Step 9: Select top SNPs and add the SNP position
res_file = 'epiDD_pair_res'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5)
```

#### (4) Parallel approximate test (recommended for big data)
Analysis can be subdivided with remma_epiAD_eff_parallel and run parallelly on different machines.


```python
# Step 1-6 is same to the above

# Step 7: parallel test. Write the codes in separate scripts and run separately.
from gmat.remma.remma_epiDD import remma_epiDD_eff_parallel
remma_epiDD_eff_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,1], 
                               var_app=var_median, p_cut=1.0e-5, out_file='epiDD_eff_parallel')
remma_epiDD_eff_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,2], 
                               var_app=var_median, p_cut=1.0e-5, out_file='epiDD_eff_parallel')
remma_epiDD_eff_parallel(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, parallel=[3,3], 
                               var_app=var_median, p_cut=1.0e-5, out_file='epiDD_eff_parallel')

# Step 8: Calculate exact p values for top SNP pairs
remma_epiDD_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="epiDD_eff_parallel.1", 
                     max_test_pair=50000, p_cut=1, out_file='epiDD_pair_res.1')
remma_epiDD_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="epiDD_eff_parallel.2", 
                     max_test_pair=50000, p_cut=1, out_file='epiDD_pair_res.2')
remma_epiDD_pair(y, xmat, zmat, gmat_lst, var_com=var_com_all, bed_file=bed_file, snp_pair_file="epiDD_eff_parallel.3", 
                     max_test_pair=50000, p_cut=1, out_file='epiDD_pair_res.3')

# Step 9: Merge files 'epiAD_pair_res.1', 'epiAD_pair_res.2' and 'epiAD_pair_res.3' 
# with the following codes.
prefix = 'epiDD_pair_res'
parallel_num = 3  # the number of parallels
with open(prefix + ".merge", 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)

# Step 10: Select top SNPs and add the SNP position
res_file = 'epiDD_pair_res.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5)
```

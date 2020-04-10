
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
  * [3.5 Exhaustive additive by additive epistatis with repeated measures](#35-Exhaustive-additive-by-additive-epistatis-with-repeated-measures)
  * [3.6 Additive test](#36-Additive-test)
  * [3.7 Dominance test](#37-Dominance-test)

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

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  
 
```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAA import remma_epiAA
from gmat.remma import annotation_snp_pos

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file) 

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
gmat_lst = [ag, ag*ag]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var_a_axa.txt')

# Step 3: Test
var_com = np.loadtxt('var_a_axa.txt')  # numpy array： [0] addtive variance; [1] additive by additive variance; [2] residual variance
remma_epiAA(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, out_file='epiAA_a_axa')

# Step 4: Select top SNPs and add the SNP position
res_file = 'epiAA_a_axa'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

#### (2) Parallel exact test  (for small data)
Analysis can be subdivided with remma_epiAA_parallel and run parallelly on different machines.

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  

```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file) 

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
gmat_lst = [ag, ag*ag]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var_a_axa.txt')

# Step 3: parallel test. Write codes of thist step in separate scripts and run parallelly

## parallel 1
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var_a_axa.txt')
ag = np.loadtxt(bed_file + '.agrm0')
gmat_lst = [ag, ag*ag]
# parallel=[3, 1] means divide total tests into three parts and run part 1
remma_epiAA_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 1], p_cut=1.0e-5, out_file='epiAA_parallel_a_axa')

## parallel 2
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var_a_axa.txt')
ag = np.loadtxt(bed_file + '.agrm0')
gmat_lst = [ag, ag*ag]
# parallel=[3, 2] means divide total tests into three parts and run part 2
remma_epiAA_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 2], p_cut=1.0e-5, out_file='epiAA_parallel_a_axa')

## parallel 3
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var_a_axa.txt')
ag = np.loadtxt(bed_file + '.agrm0')
gmat_lst = [ag, ag*ag]
# parallel=[3, 3] means divide total tests into three parts and run part 3
remma_epiAA_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 3], p_cut=1.0e-5, out_file='epiAA_parallel_a_axa')

# Step 4: Merge files 'epiAA_parallel_a_axa.*' with the following codes.
import os
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
        os.remove(prefix + '.' + str(i))

# Step 5: Select top SNPs and add the SNP position
from gmat.remma import annotation_snp_pos
res_file = 'epiAA_parallel_a_axa.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0

```

</code></pre>
</details>

#### (3) approximate test (recommended for big data)

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  

```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAA import remma_epiAA_approx
from gmat.remma import annotation_snp_pos

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file) 

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
gmat_lst = [ag, ag*ag]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var_a_axa.txt')

# Step 3: Approximate test
var_com = np.loadtxt('var_a_axa.txt')  # numpy array： [0] addtive variance; [1] additive by additive variance; [2] residual variance
remma_epiAA_approx(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, num_random_pair=100000, out_file='epiAA_approx_a_axa')

# Step 4: Select top SNPs and add the SNP position
res_file = 'epiAA_approx_a_axa'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

#### (4) Parallel approximate test (recommended for big data)
Analysis can be subdivided with remma_epiAA_approx_parallel and run parallelly on different machines.

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  

```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file) 

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
gmat_lst = [ag, ag*ag]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var_a_axa.txt')

# Step 3: parallel approximate test. Write codes of thist step in separate scripts and run parallelly

## parallel 1
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var_a_axa.txt')
ag = np.loadtxt(bed_file + '.agrm0')
gmat_lst = [ag, ag*ag]
# parallel=[3, 1] means divide total tests into three parts and run part 1
remma_epiAA_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 1], p_cut=1.0e-5, out_file='epiAA_approx_parallel_a_axa')

## parallel 2
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var_a_axa.txt')
ag = np.loadtxt(bed_file + '.agrm0')
gmat_lst = [ag, ag*ag]
# parallel=[3, 2] means divide total tests into three parts and run part 2
remma_epiAA_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 2], p_cut=1.0e-5, out_file='epiAA_approx_parallel_a_axa')

## parallel 3
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var_a_axa.txt')
ag = np.loadtxt(bed_file + '.agrm0')
gmat_lst = [ag, ag*ag]
# parallel=[3, 3] means divide total tests into three parts and run part 3
remma_epiAA_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 3], p_cut=1.0e-5, out_file='epiAA_approx_parallel_a_axa')

# Step 4: Merge files 'epiAA_approx_parallel_a_axa.*' 
# with the following codes.
import os
prefix = 'epiAA_approx_parallel_a_axa'
parallel_num = 3  # the number of parallels
with open(prefix + '.merge', 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)
        os.remove(prefix + '.' + str(i))

# Step 5: Select top SNPs and add the SNP position
from gmat.remma import annotation_snp_pos                   
res_file = 'epiAA_approx_parallel_a_axa.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```
</code></pre>
</details>

### 3.2.2 Include additive, dominance and additive by additive genomic relationship matrix
#### (1) Exact test (for small data)

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAA import remma_epiAA
from gmat.remma import annotation_snp_pos

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var_a_d_axa.txt')

# Step 3: Test
var_com = np.loadtxt('var_a_d_axa.txt')  # numpy array： [0] addtive variance; [1] dominance variance; [2] additive by additive variance; [3] residual variance
remma_epiAA(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, out_file='epiAA_a_d_axa')

# Step 4: Select top SNPs and add the SNP position
res_file = 'epiAA_a_d_axa'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0

```

</code></pre>
</details>

#### (2) Parallel exact test (for small data)
Analysis can be subdivided with remma_epiAA_parallel and run parallelly on different machines.

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  

```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var_a_d_axa.txt')

# Step 3: parallel test. Write codes of thist step in separate scripts and run parallelly

## parallel 1
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var_a_d_axa.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag] 
# parallel=[3, 1] means divide total tests into three parts and run part 1
remma_epiAA_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 1], p_cut=1.0e-5, out_file='epiAA_parallel_a_d_axa')

## parallel 2
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var_a_d_axa.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag] 
# parallel=[3, 2] means divide total tests into three parts and run part 2
remma_epiAA_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 2], p_cut=1.0e-5, out_file='epiAA_parallel_a_d_axa')

## parallel 3
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var_a_d_axa.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag] 
# parallel=[3, 3] means divide total tests into three parts and run part 3
remma_epiAA_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 3], p_cut=1.0e-5, out_file='epiAA_parallel_a_d_axa')

# Step 4: Merge files 'epiAA_parallel_a_d_axa.*' with the following codes.
import os
prefix = 'epiAA_parallel_a_d_axa'
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
        os.remove(prefix + '.' + str(i))

# Step 5: Select top SNPs and add the SNP position
from gmat.remma import annotation_snp_pos
res_file = 'epiAA_parallel_a_d_axa.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```
</code></pre>
</details>

#### (3) approximate test (recommended for big data)

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAA import remma_epiAA_approx
from gmat.remma import annotation_snp_pos

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var_a_d_axa.txt')

# Step 3: Approximate test
var_com = np.loadtxt('var_a_d_axa.txt')  # numpy array： [0] addtive variance; [1] dominance variance; [2] additive by additive variance; [3] residual variance
remma_epiAA_approx(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, num_random_pair=100000, out_file='epiAA_approx_a_d_axa')

# Step 4: Select top SNPs and add the SNP position
res_file = 'epiAA_approx_a_d_axa'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

#### (4) Parallel approximate test (recommended for big data)
Analysis can be subdivided with remma_epiAA_approx_parallel and run parallelly on different machines.

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file) 
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var_a_d_axa.txt')

# Step 3: parallel approximate test. Write codes of thist step in separate scripts and run parallelly

## parallel 1
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var_a_d_axa.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag]
# parallel=[3, 1] means divide total tests into three parts and run part 1
remma_epiAA_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 1], p_cut=1.0e-5, out_file='epiAA_approx_parallel_a_d_axa')

## parallel 2
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var_a_d_axa.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag]
# parallel=[3, 2] means divide total tests into three parts and run part 2
remma_epiAA_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 2], p_cut=1.0e-5, out_file='epiAA_approx_parallel_a_d_axa')

## parallel 3
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var_a_d_axa.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag]
# parallel=[3, 3] means divide total tests into three parts and run part 3
remma_epiAA_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 3], p_cut=1.0e-5, out_file='epiAA_approx_parallel_a_d_axa')

# Step 4: Merge files 'epiAA_approx_parallel_a_d_axa.*' 
# with the following codes.
import os
prefix = 'epiAA_approx_parallel_a_d_axa'
parallel_num = 3  # the number of parallels
with open(prefix + '.merge', 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)
        os.remove(prefix + '.' + str(i))

# Step 5: Select top SNPs and add the SNP position
from gmat.remma import annotation_snp_pos                   
res_file = 'epiAA_approx_parallel_a_d_axa.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

### 3.2.3 Include additive, dominance and three kinds of epistatic genomic relationship matrix
additive, dominance, additive by additive, additive by dominance and dominance by dominance genomic relationship matrix   
#### (1) Exact test (for small data)

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAA import remma_epiAA
from gmat.remma import annotation_snp_pos

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: Test
var_com = np.loadtxt('var.txt') # numpy array： [0] addtive variance; [1] dominance variance; [2] additive by additive variance; 
                                #               [3] additive by dominance variance; [4] dominance by dominance variance; [5] residual variance
remma_epiAA(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, out_file='epiAA')

# Step 4: Select top SNPs and add the SNP position
res_file = 'epiAA'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

#### (2) Parallel exact test (for small data)
Analysis can be subdivided with remma_epiAA_parallel and run parallelly on different machines.

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: parallel test. Write codes of thist step in separate scripts and run parallelly

## parallel 1
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg] 
# parallel=[3, 1] means divide total tests into three parts and run part 1
remma_epiAA_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 1], p_cut=1.0e-5, out_file='epiAA_parallel')

## parallel 2
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg] 
# parallel=[3, 2] means divide total tests into three parts and run part 2
remma_epiAA_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 2], p_cut=1.0e-5, out_file='epiAA_parallel')

## parallel 3
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg] 
# parallel=[3, 3] means divide total tests into three parts and run part 3
remma_epiAA_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 3], p_cut=1.0e-5, out_file='epiAA_parallel')

# Step 4: Merge files 'epiAA_parallel_a_d_axa.*' with the following codes.
import os
prefix = 'epiAA_parallel'
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
        os.remove(prefix + '.' + str(i))

# Step 5: Select top SNPs and add the SNP position
from gmat.remma import annotation_snp_pos
res_file = 'epiAA_parallel.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

#### (3) approximate test (recommended for big data)

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAA import remma_epiAA_approx
from gmat.remma import annotation_snp_pos

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: Approximate test
var_com = np.loadtxt('var.txt')  # numpy array： [0] addtive variance; [1] dominance variance; [2] additive by additive variance; [3] residual variance
remma_epiAA_approx(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, num_random_pair=100000, out_file='epiAA_approx')

# Step 4: Select top SNPs and add the SNP position
res_file = 'epiAA_approx'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0

```

</code></pre>
</details>

#### (4) Parallel approximate test (recommended for big data)
Analysis can be subdivided with remma_epiAA_approx_parallel and run parallelly on different machines.

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file) 
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: parallel approximate test. Write codes of thist step in separate scripts and run parallelly

## parallel 1
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]
# parallel=[3, 1] means divide total tests into three parts and run part 1
remma_epiAA_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 1], p_cut=1.0e-5, out_file='epiAA_approx_parallel')

## parallel 2
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]
# parallel=[3, 2] means divide total tests into three parts and run part 2
remma_epiAA_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 2], p_cut=1.0e-5, out_file='epiAA_approx_parallel')

## parallel 3
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]
# parallel=[3, 3] means divide total tests into three parts and run part 3
remma_epiAA_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 3], p_cut=1.0e-5, out_file='epiAA_approx_parallel')

# Step 4: Merge files 'epiAA_approx_parallel_a_d_axa.*' 
# with the following codes.
import os
prefix = 'epiAA_approx_parallel'
parallel_num = 3  # the number of parallels
with open(prefix + '.merge', 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)
        os.remove(prefix + '.' + str(i))

# Step 5: Select top SNPs and add the SNP position
from gmat.remma import annotation_snp_pos                   
res_file = 'epiAA_approx_parallel.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

## 3.3 Exhaustive additive by dominance epistatis  
Data: Mouse data in directory of GMAT/examples/data/mouse  
Include additive, dominance, additive by additive, additive by dominance and dominance by dominance genomic relationship matrix  
#### (1) Exact test (for small data)

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAD import remma_epiAD
from gmat.remma import annotation_snp_pos

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: Test
var_com = np.loadtxt('var.txt') # numpy array： [0] addtive variance; [1] dominance variance; [2] additive by additive variance; 
                                #               [3] additive by dominance variance; [4] dominance by dominance variance; [5] residual variance
remma_epiAD(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, out_file='epiAD')

# Step 4: Select top SNPs and add the SNP position
res_file = 'epiAD'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0

```

</code></pre>
</details>

#### (2) Parallel exact test (for small data)
Analysis can be subdivided with remma_epiAD_parallel and run parallelly on different machines.

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: parallel test. Write codes of thist step in separate scripts and run parallelly

## parallel 1
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAD import remma_epiAD_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg] 
# parallel=[3, 1] means divide total tests into three parts and run part 1
remma_epiAD_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 1], p_cut=1.0e-5, out_file='epiAD_parallel')

## parallel 2
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAD import remma_epiAD_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg] 
# parallel=[3, 2] means divide total tests into three parts and run part 2
remma_epiAD_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 2], p_cut=1.0e-5, out_file='epiAD_parallel')

## parallel 3
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAD import remma_epiAD_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg] 
# parallel=[3, 3] means divide total tests into three parts and run part 3
remma_epiAD_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 3], p_cut=1.0e-5, out_file='epiAD_parallel')

# Step 4: Merge files 'epiAD_parallel.*' with the following codes.
import os
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
        os.remove(prefix + '.' + str(i))

# Step 5: Select top SNPs and add the SNP position
from gmat.remma import annotation_snp_pos
res_file = 'epiAD_parallel.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

#### (3) approximate test (recommended for big data)

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAD import remma_epiAD_approx
from gmat.remma import annotation_snp_pos

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: Approximate test
var_com = np.loadtxt('var.txt')  # numpy array： [0] addtive variance; [1] dominance variance; [2] additive by additive variance; [3] residual variance
remma_epiAD_approx(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, num_random_pair=100000, out_file='epiAD_approx')

# Step 4: Select top SNPs and add the SNP position
res_file = 'epiAD_approx'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

#### (4) Parallel approximate test (recommended for big data)
Analysis can be subdivided with remma_epiAD_approx_parallel and run parallelly on different machines.

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file) 
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: parallel approximate test. Write codes of thist step in separate scripts and run parallelly

## parallel 1
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAD import remma_epiAD_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]
# parallel=[3, 1] means divide total tests into three parts and run part 1
remma_epiAD_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 1], p_cut=1.0e-5, out_file='epiAD_approx_parallel')

## parallel 2
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAD import remma_epiAD_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]
# parallel=[3, 2] means divide total tests into three parts and run part 2
remma_epiAD_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 2], p_cut=1.0e-5, out_file='epiAD_approx_parallel')

## parallel 3
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAD import remma_epiAD_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]
# parallel=[3, 3] means divide total tests into three parts and run part 3
remma_epiAD_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 3], p_cut=1.0e-5, out_file='epiAD_approx_parallel')

# Step 4: Merge files 'epiAA_approx_parallel.*' 
# with the following codes.
import os
prefix = 'epiAD_approx_parallel'
parallel_num = 3  # the number of parallels
with open(prefix + '.merge', 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)
        os.remove(prefix + '.' + str(i))

# Step 5: Select top SNPs and add the SNP position
from gmat.remma import annotation_snp_pos                   
res_file = 'epiAD_approx_parallel.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

## 3.4 Exhaustive dominance by dominance epistatis  
Data: Mouse data in directory of GMAT/examples/data/mouse  
Include additive, dominance, additive by additive, additive by dominance and dominance by dominance genomic relationship matrix   
#### (1) Exact test (for small data)

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  

```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiDD import remma_epiDD
from gmat.remma import annotation_snp_pos

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: Test
var_com = np.loadtxt('var.txt') # numpy array： [0] addtive variance; [1] dominance variance; [2] additive by additive variance; 
                                #               [3] additive by dominance variance; [4] dominance by dominance variance; [5] residual variance
remma_epiDD(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, out_file='epiDD')

# Step 4: Select top SNPs and add the SNP position
res_file = 'epiDD'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

#### (2) Parallel exact test (for small data)
Analysis can be subdivided with remma_epiDD_parallel and run parallelly on different machines.

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: parallel test. Write codes of thist step in separate scripts and run parallelly

## parallel 1
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiDD import remma_epiDD_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg] 
# parallel=[3, 1] means divide total tests into three parts and run part 1
remma_epiDD_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 1], p_cut=1.0e-5, out_file='epiDD_parallel')

## parallel 2
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiDD import remma_epiDD_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg] 
# parallel=[3, 2] means divide total tests into three parts and run part 2
remma_epiDD_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 2], p_cut=1.0e-5, out_file='epiDD_parallel')

## parallel 3
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiDD import remma_epiDD_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg] 
# parallel=[3, 3] means divide total tests into three parts and run part 3
remma_epiDD_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 3], p_cut=1.0e-5, out_file='epiDD_parallel')

# Step 4: Merge files 'epiDD_parallel.*' with the following codes.
import os
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
        os.remove(prefix + '.' + str(i))

# Step 5: Select top SNPs and add the SNP position
from gmat.remma import annotation_snp_pos
res_file = 'epiDD_parallel.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

#### (3) approximate test (recommended for big data)

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  

```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiDD import remma_epiDD_approx
from gmat.remma import annotation_snp_pos

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: Approximate test
var_com = np.loadtxt('var.txt')  # numpy array： [0] addtive variance; [1] dominance variance; [2] additive by additive variance; [3] residual variance
remma_epiDD_approx(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, num_random_pair=100000, out_file='epiDD_approx')

# Step 4: Select top SNPs and add the SNP position
res_file = 'epiDD_approx'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

#### (4) Parallel approximate test (recommended for big data)
Analysis can be subdivided with remma_epiDD_approx_parallel and run parallelly on different machines.

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>

```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file) 
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: parallel approximate test. Write codes of thist step in separate scripts and run parallelly

## parallel 1
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiDD import remma_epiDD_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]
# parallel=[3, 1] means divide total tests into three parts and run part 1
remma_epiDD_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 1], p_cut=1.0e-5, out_file='epiDD_approx_parallel')

## parallel 2
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiDD import remma_epiDD_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]
# parallel=[3, 2] means divide total tests into three parts and run part 2
remma_epiDD_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 2], p_cut=1.0e-5, out_file='epiDD_approx_parallel')

## parallel 3
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiDD import remma_epiDD_approx_parallel
bed_file = 'plink'
pheno_file = 'pheno'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
dg = np.loadtxt(bed_file + '.dgrm_as0')
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]
# parallel=[3, 3] means divide total tests into three parts and run part 3
remma_epiDD_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 3], p_cut=1.0e-5, out_file='epiDD_approx_parallel')

# Step 4: Merge files 'epiDD_approx_parallel.*' 
# with the following codes.
import os
prefix = 'epiDD_approx_parallel'
parallel_num = 3  # the number of parallels
with open(prefix + '.merge', 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)
        os.remove(prefix + '.' + str(i))

# Step 5: Select top SNPs and add the SNP position
from gmat.remma import annotation_snp_pos                   
res_file = 'epiDD_approx_parallel.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```
</code></pre>
</details>

## 3.5 Exhaustive additive by additive epistatis with repeated measures
Data: Yeast data in directory of GMAT/examples/data/yeast   
No heterozygous genotypes. No dominance effects.   
  
#### (1) Exact test (for small data)

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAA import remma_epiAA
from gmat.remma import annotation_snp_pos

# Step 1: Calculate the genomic relationship matrix
bed_file = 'CobaltChloride'  # the prefix for the plink binary file
agmat(bed_file) 

# Step 2: Estimate the variances
pheno_file = 'CobaltChloride'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
pe = np.eye(ag.shape[0])  # identity matrix with dimension equal to the number of individuals. 
                          # to model the individual-specific error (permanent environmental effect)
gmat_lst = [ag, ag*ag, pe]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: Test
var_com = np.loadtxt('var.txt')  # numpy array： [0] addtive variance; [1] additive by additive variance; 
                                             # [2] individual-specific error variance [3] residual variance
remma_epiAA(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, out_file='epiAA')

# Step 4: Select top SNPs and add the SNP position
res_file = 'epiAA'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

#### (2) Parallel exact test (for small data)
Analysis can be subdivided with remma_epiAA_parallel and run parallelly on different machines.

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code> 

```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat
from gmat.uvlmm.design_matrix import design_matrix_wemai_multi_gmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat

# Step 1: Calculate the genomic relationship matrix
bed_file = 'CobaltChloride'  # the prefix for the plink binary file
agmat(bed_file) 

# Step 2: Estimate the variances
pheno_file = 'CobaltChloride'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
pe = np.eye(ag.shape[0])  # identity matrix with dimension equal to the number of individuals. 
                          # to model the individual-specific error (permanent environmental effect)
gmat_lst = [ag, ag*ag, pe]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')


# Step 3: parallel test. Write codes of thist step in separate scripts and run parallelly

## parallel 1
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_parallel
bed_file = 'CobaltChloride'
pheno_file = 'CobaltChloride'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
pe = np.eye(ag.shape[0]) 
gmat_lst = [ag, ag*ag, pe]
# parallel=[3, 1] means divide total tests into three parts and run part 1
remma_epiAA_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 1], p_cut=1.0e-5, out_file='epiAA_parallel')

## parallel 2
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_parallel
bed_file = 'CobaltChloride'
pheno_file = 'CobaltChloride'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
pe = np.eye(ag.shape[0]) 
gmat_lst = [ag, ag*ag, pe]
# parallel=[3, 2] means divide total tests into three parts and run part 2
remma_epiAA_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 2], p_cut=1.0e-5, out_file='epiAA_parallel')

## parallel 3
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_parallel
bed_file = 'CobaltChloride'
pheno_file = 'CobaltChloride'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
pe = np.eye(ag.shape[0]) 
gmat_lst = [ag, ag*ag, pe]
# parallel=[3, 3] means divide total tests into three parts and run part 3
remma_epiAA_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 3], p_cut=1.0e-5, out_file='epiAA_parallel')

# Step 4: Merge files 'epiAA_parallel.*' with the following codes.
import os
prefix = 'epiAA_parallel'
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
        os.remove(prefix + '.' + str(i))

# Step 5: Select top SNPs and add the SNP position
from gmat.remma import annotation_snp_pos
res_file = 'epiAA_parallel.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=1.0e-5, dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

#### (3) approximate test (recommended for big data)

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code>  


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_epiAA import remma_epiAA_approx
from gmat.remma import annotation_snp_pos

# Step 1: Calculate the genomic relationship matrix
bed_file = 'CobaltChloride'  # the prefix for the plink binary file
agmat(bed_file) 

# Step 2: Estimate the variances
pheno_file = 'CobaltChloride'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
pe = np.eye(ag.shape[0])  # identity matrix with dimension equal to the number of individuals. 
                          # to model the individual-specific error (permanent environmental effect)
gmat_lst = [ag, ag*ag, pe]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: Approximate test
var_com = np.loadtxt('var.txt')  # numpy array： [0] addtive variance; [1] additive by additive variance; 
                                             # [2] individual-specific error variance [3] residual variance
remma_epiAA_approx(pheno_file, bed_file, gmat_lst, var_com, p_cut=1.0e-5, num_random_pair=100000, out_file='epiAA_approx')

# Step 4: Select top SNPs and add the SNP position
res_file = 'epiAA_approx'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=0.05*2/(28220*28219), dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0
```

</code></pre>
</details>

#### (4) Parallel approximate test (recommended for big data)

Analysis can be subdivided with remma_epiAA_approx_parallel and run parallelly on different machines.
<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code> 

```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd
from gmat.gmatrix import agmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat

# Step 1: Calculate the genomic relationship matrix
bed_file = 'CobaltChloride'  # the prefix for the plink binary file
agmat(bed_file) 

# Step 2: Estimate the variances
pheno_file = 'CobaltChloride'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
pe = np.eye(ag.shape[0])  # identity matrix with dimension equal to the number of individuals. 
                          # to model the individual-specific error (permanent environmental effect)
gmat_lst = [ag, ag*ag, pe]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var.txt')

# Step 3: parallel approximate test. Write codes of thist step in separate scripts and run parallelly

## parallel 1
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_approx_parallel
bed_file = 'CobaltChloride'
pheno_file = 'CobaltChloride'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
pe = np.eye(ag.shape[0]) 
gmat_lst = [ag, ag*ag, pe]
# parallel=[3, 1] means divide total tests into three parts and run part 1
remma_epiAA_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 1], p_cut=1.0e-5, out_file='epiAA_approx_parallel')

## parallel 2
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_approx_parallel
bed_file = 'CobaltChloride'
pheno_file = 'CobaltChloride'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
pe = np.eye(ag.shape[0]) 
gmat_lst = [ag, ag*ag, pe]
# parallel=[3, 2] means divide total tests into three parts and run part 2
remma_epiAA_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 2], p_cut=1.0e-5, out_file='epiAA_approx_parallel')

## parallel 3
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.remma.remma_epiAA import remma_epiAA_approx_parallel
bed_file = 'CobaltChloride'
pheno_file = 'CobaltChloride'
var_com = np.loadtxt('var.txt')
ag = np.loadtxt(bed_file + '.agrm0')
pe = np.eye(ag.shape[0]) 
gmat_lst = [ag, ag*ag, pe]
# parallel=[3, 3] means divide total tests into three parts and run part 3
remma_epiAA_approx_parallel(pheno_file, bed_file, gmat_lst, var_com, parallel=[3, 3], p_cut=1.0e-5, out_file='epiAA_approx_parallel')

# Step 4: Merge files 'epiAA_approx_parallel.*' 
# with the following codes.
import os
prefix = 'epiAA_approx_parallel'
parallel_num = 3  # the number of parallels
with open(prefix + '.merge', 'w') as fout:
    with open(prefix + '.1') as fin:
        head_line = fin.readline()
        fout.write(head_line)
    for i in range(1, parallel_num+1):
        with open(prefix + '.' + str(i)) as fin:
            head_line = fin.readline()
            for line in fin:
                fout.write(line)
        os.remove(prefix + '.' + str(i))

# Step 5: Select top SNPs and add the SNP position
from gmat.remma import annotation_snp_pos                   
res_file = 'epiAA_approx_parallel.merge'  # result file
annotation_snp_pos(res_file, bed_file, p_cut=0.05*2/(28220*28219), dis=0)  # p values < 1.0e-5 and the distance between SNP pairs > 0

```

</code></pre>
</details>


## 3.6 Additive test
#### No repeated measures
Data: Mouse data in directory of GMAT/examples/data/mouse

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code> 

```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma import remma_add

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # The first one must be addtive genomic relationship matrix. The others can be removed from the list.
                                          # ag*ag: additive by additive; ag*dg: additive by dominance; dg*dg: dominance by dominance
# gmat_lst = [ag]
# gmat_lst = [ag, dg]
# gmat_lst = [ag, dg, ag*ag]
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var_add.txt')

# Step 3: Test
var_com = np.loadtxt('var_add.txt') # numpy array
res = remma_add(pheno_file, bed_file, gmat_lst, var_com, out_file='remma_add')
```

</code></pre>
</details>

#### With  repeated measures
Data: Yeast data in directory of GMAT/examples/data/yeast
No heterozygous genotypes. No dominance effects.

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code> 

```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma.remma_add import remma_add

# Step 1: Calculate the genomic relationship matrix
bed_file = 'CobaltChloride'  # the prefix for the plink binary file
agmat(bed_file) 

# Step 2: Estimate the variances
pheno_file = 'CobaltChloride'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
pe = np.eye(ag.shape[0])  # identity matrix with dimension equal to the number of individuals. 
                          # to model the individual-specific error (permanent environmental effect)
gmat_lst = [ag, ag*ag, pe]  # ag*ag is the additive by additive genomic relationship matrix
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var_add.txt')

# Step 3: Test
var_com = np.loadtxt('var_add.txt')  # numpy array： [0] addtive variance; [1] additive by additive variance; 
                                             # [2] individual-specific error variance [3] residual variance
res = remma_add(pheno_file, bed_file, gmat_lst, var_com, out_file='remma_add')
```

</code></pre>
</details>

## 3.7 Dominance test
Data: Mouse data in directory of GMAT/examples/data/mouse

<details>
  <summary><mark><font color=red>Click to view codes</font></mark></summary
  <pre><code> 


```python
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from gmat.gmatrix import agmat, dgmat_as
from gmat.uvlmm.uvlmm_varcom import wemai_multi_gmat
from gmat.remma import remma_dom

# Step 1: Calculate the genomic relationship matrix
bed_file = 'plink'  # the prefix for the plink binary file
agmat(bed_file)  # additive genomic relationship matrix
dgmat_as(bed_file)  # dominance genomic relationship matrix

# Step 2: Estimate the variances
pheno_file = 'pheno'  # phenotypic file
ag = np.loadtxt(bed_file + '.agrm0')  # load the additive genomic relationship matrix
dg = np.loadtxt(bed_file + '.dgrm_as0')  # load the dominance genomic relationship matrix
gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg]  # The first one must be addtive genomic relationship matrix. 
                                          # The second one must be dominance genomic relationship matrix. 
                                          # The others can be removed from the list.
                                          # ag*ag: additive by additive; ag*dg: additive by dominance; dg*dg: dominance by dominance
# gmat_lst = [ag, dg]
# gmat_lst = [ag, dg, ag*ag]
# gmat_lst = [ag, dg, ag*ag, ag*dg, dg*dg, np.eye(ag.shape[0])]  # Not fit this data, but can be used for the other data with repeated mesures.
wemai_multi_gmat(pheno_file, bed_file, gmat_lst, out_file='var_dom.txt')

# Step 3: Test
var_com = np.loadtxt('var_dom.txt') # numpy array
res = remma_dom(pheno_file, bed_file, gmat_lst, var_com, out_file='remma_dom')
```


</code></pre>
</details>
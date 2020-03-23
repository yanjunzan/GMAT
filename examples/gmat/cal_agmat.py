"""
Calculate the additive genomic relationship matrix


"""

import numpy as np
import pandas as pd
import logging
import os
from gmat.gmatrix import agmat, dgmat_as

logging.basicConfig(level=logging.INFO)

bed_file = '../data/mouse/plink'

# matrix form
agmat0 = agmat(bed_file, inv=True, small_val=0.001, out_fmt='mat')
grm0 = np.load('../data/mouse/plink.agrm0.npz')
giv0 = np.load('../data/mouse/plink.agiv0.npz')
np.savetxt('../data/mouse/plink.agrm0', grm0['mat'])
np.savetxt('../data/mouse/plink.agiv0', giv0['mat'])



# row-column-value form, which can be used by asreml
agmat1 = agmat(bed_file, inv=True, small_val=0.001, out_fmt='row_col_val')
grm1 = np.load('../data/mouse/plink.agrm1.npz')
giv1 = np.load('../data/mouse/plink.agiv1.npz')
grm1_dct = {'row': grm1['row'] + 1,
            'col': grm1['col'] + 1,
            'val': grm1['val']
}
grm1_df = pd.DataFrame(grm1_dct, columns=['row', 'col', 'val'])
grm1_df.to_csv('../data/mouse/plink.agrm1', header=False, index=False, sep=' ')
giv1_dct = {'row': giv1['row'] + 1,
            'col': giv1['col'] + 1,
            'val': giv1['val']
}
giv1_df = pd.DataFrame(giv1_dct, columns=['row', 'col', 'val'])
giv1_df.to_csv('../data/mouse/plink.agiv1', header=False, index=False, sep=' ')



# id-id-value form
agmat2 = agmat(bed_file, inv=True, small_val=0.001, out_fmt='id_id_val')
grm2 = np.load('../data/mouse/plink.agrm2.npz')
giv2 = np.load('../data/mouse/plink.agiv2.npz')

grm2_dct = {'id0': grm2['id0'],
            'id1': grm2['id1'],
            'val': grm2['val']
}
grm2_df = pd.DataFrame(grm2_dct, columns=['id0', 'id1', 'val'])
grm2_df.to_csv('../data/mouse/plink.agrm2', header=False, index=False, sep=' ')
giv2_dct = {'id0': giv2['id0'],
            'id1': giv2['id1'],
            'val': giv2['val']
}
giv2_df = pd.DataFrame(giv2_dct, columns=['id0', 'id1', 'val'])
giv2_df.to_csv('../data/mouse/plink.agiv2', header=False, index=False, sep=' ')


# delete

os.remove('../data/mouse/plink.agrm0.npz')
os.remove('../data/mouse/plink.agiv0.npz')
os.remove('../data/mouse/plink.agrm0')
os.remove('../data/mouse/plink.agiv0')

os.remove('../data/mouse/plink.agrm1.npz')
os.remove('../data/mouse/plink.agiv1.npz')
os.remove('../data/mouse/plink.agrm1')
os.remove('../data/mouse/plink.agiv1')

os.remove('../data/mouse/plink.agrm2.npz')
os.remove('../data/mouse/plink.agiv2.npz')
os.remove('../data/mouse/plink.agrm2')
os.remove('../data/mouse/plink.agiv2')


import numpy as np
import pandas as pd
from _cread_plink_bed import ffi, lib


class Bed():
    """
    描述：
    读入plink的bed文件
    """

    def __init__(self, bed_file):
        self.bed_file = bed_file
        self.snp_info = pd.read_csv(bed_file + '.bim', header=None, sep='\s+')
        self.num_snp = self.snp_info.shape[0]
        self.id_info = pd.read_csv(bed_file + '.fam', header=None, sep='\s+')
        self.num_id = self.id_info.shape[0]

    def read(self):
        pbed_file = ffi.new("char[]", self.bed_file.encode('ascii'))
        pnum_id = ffi.cast("long long", self.num_id)
        pnum_snp = ffi.cast("long long", self.num_snp)
        marker_mat = np.ones(self.num_id * self.num_snp, dtype=np.float64)
        pmarker_mat = ffi.cast("double *", ffi.from_buffer(marker_mat))
        lib.read_plink_bed(pbed_file, pnum_id, pnum_snp, pmarker_mat)
        marker_mat[np.abs(marker_mat - 1.0/3) < 0.0001] = np.nan
        marker_mat.shape = self.num_snp, self.num_id
        return marker_mat.T


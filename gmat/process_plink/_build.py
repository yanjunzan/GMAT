# ext_build.py

from os.path import dirname, join, realpath
import cffi

ffi = cffi.FFI() #生成cffi实例

ffi.cdef("""int read_plink_bed(char *bed_file, long long num_id, long long num_snp, double *marker_mat);""") #函数声明

ffi.set_source('_cread_plink_bed', """
    int read_plink_bed(char *bed_file, long long num_id, long long num_snp, double *marker_mat);
""",
sources=[join("gmat/process_plink", "_read_plink_bed.c")],
)

if __name__ == '__main__':
    ffi.compile(verbose=True)


'''
from os.path import dirname, join, realpath
import cffi

ffi = cffi.FFI() #生成cffi实例

ffi.cdef("""int read_plink_bed(char *bed_file, long long num_id, long long num_snp, double *marker_mat);""") #函数声明

ffi.set_source('_cread_plink_bed', """
    int read_plink_bed(char *bed_file, long long num_id, long long num_snp, double *marker_mat);
""",
sources=[join("gmat/process_plink", "_read_plink_bed.cpp")],
source_extension='.cpp'
)

if __name__ == '__main__':
    ffi.compile(verbose=True)
'''
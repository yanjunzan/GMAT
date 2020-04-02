# ext_build.py

from os.path import dirname, join, realpath
import cffi

ffi = cffi.FFI() #生成cffi实例

ffi.cdef("""int remma_epiAA_eff_cpu(char *bed_file, long long num_id, long long num_snp, long long *snp_lst_0, 
    long long len_snp_lst_0, double *pymat, double eff_cut, char* out_file);""")  # 函数声明

ffi.cdef("""int remma_epiAD_eff_cpu(char *bed_file, long long num_id, long long num_snp, 
     long long *snp_lst_0, long long len_snp_lst_0, double *pymat, double eff_cut, char* out_file);""")  # 函数声明

ffi.cdef("""int remma_epiDD_eff_cpu(char *bed_file, long long num_id, long long num_snp, 
     long long *snp_lst_0, long long len_snp_lst_0, double *pymat, double eff_cut, char* out_file);""")  # 函数声明

ffi.set_source('_cremma_epi_eff_cpu', """
    int remma_epiAA_eff_cpu(char *bed_file, long long num_id, long long num_snp, long long *snp_lst_0, 
        long long len_snp_lst_0, double *pymat, double eff_cut, char* out_file);
    int remma_epiAD_eff_cpu(char *bed_file, long long num_id, long long num_snp, 
        long long *snp_lst_0, long long len_snp_lst_0, double *pymat, double eff_cut, char* out_file);
    int remma_epiDD_eff_cpu(char *bed_file, long long num_id, long long num_snp, 
     long long *snp_lst_0, long long len_snp_lst_0, double *pymat, double eff_cut, char* out_file);
""",
sources=[join("gmat/remma", "_remma_epi_eff_cpu.c")],
extra_compile_args = ['-fopenmp', '-lm'],
extra_link_args = ['-fopenmp', '-lm'],
)


if __name__ == '__main__':
    ffi.compile(verbose=True)


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h> 
#include <time.h>
//#define CLOCKS_PER_SEC ((clock_t)1000)


int read_plink_bed(char *bed_file, long long num_id, long long num_snp, double *marker_mat)
{
	/*读取bed文件，生成标记矩阵*/
	//打开bed file
	char in_file[1000];
	strcpy(in_file, bed_file);
	strcat(in_file, ".bed");
	FILE *fin_bed = fopen(in_file, "r");
	if(fin_bed==NULL){
		printf("Fail to open the plink bed file: %s.\n", in_file);
		exit(1);
	}
	
	//读取文件
	long long num_block = num_id/4 + 1; //存储一个位点所有SNP占据的block（一个字节），每个字节可存储4个SNP 
	long long num_snp_last = num_id % 4; //最后一个block存储的SNP数 
	if(num_snp_last == 0){
		num_snp_last = 4;
		num_block = num_block - 1;
	}
	
	//按顺序读取每个字节
	long long i = 0, m = 0, k = 0;
	char x03 = '3' - 48;
	unsigned char one_byte;
	int code_val;
	fseek(fin_bed, 3, SEEK_SET); //跳过开头三个字节 
	while(fread(&one_byte, sizeof(char), 1, fin_bed) == 1){
		i++;
		if(i % num_block != 0){
			for(m = 0; m < 4; m++){
				code_val = (one_byte >> (2*m)) & x03;
				marker_mat[k++] = (code_val*code_val + code_val)/6.0;
			}
		}
		else{
			for(m = 0; m < num_snp_last; m++){
				code_val = (one_byte >> (2*m)) & x03;
				marker_mat[k++] = (code_val*code_val + code_val)/6.0;
			}
		}
		
	}
	fclose(fin_bed);
	fin_bed=NULL;
	return 1;
}


/***********加加互作 ***********/ 

int print_outAA(long long i, long long *snp_lst_0, long long num_snp, long long num_id, double *marker_mat, double *pymat, 
        double eff_cut, FILE *out_res, long long len_snp_lst_0, char *bar) {
	/*输出互作效应，为openmp服务 */
	long long j=0, k=0;
	double epi_effect=0.0;
	clock_t start, finish;
	double  duration;
	start = clock();
	for(j = snp_lst_0[i]+1; j < num_snp; j++){
		epi_effect = 0.0;
	    for(k = 0; k < num_id; k++){
			epi_effect += marker_mat[snp_lst_0[i]*num_id + k] * marker_mat[j*num_id + k] * pymat[k];
		}
		if(fabs(epi_effect) > eff_cut){
        	fprintf(out_res, "%lld %lld %g\n", snp_lst_0[i], j, epi_effect);
    	}
	}
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	int bar_len = (int) i*100.0/len_snp_lst_0 + 1;
	char bari[102];
	strncpy(bari, bar, bar_len);
	printf("\r[%-101s] [%3d%%] [consuming time: %g seconds]", bari, bar_len, duration);
	fflush(stdout);
	return 0;
}




int remma_epiAA_eff_cpu(char *bed_file, long long num_id, long long num_snp, 
     long long *snp_lst_0, long long len_snp_lst_0, double *pymat, double eff_cut, char* out_file)
{
	/*加加上位*/ 
	//标记矩阵声明空间
	double *marker_mat = (double*) calloc(num_id*num_snp, sizeof(double));
	read_plink_bed(bed_file, num_id, num_snp, marker_mat);
	
	//标记矩阵中心化，-2p 
	long long i = 0, j = 0;
    double pFreq = 0.0;// frequence of one allele for each SNP
    //double scale = 0;// scale factor
    for(i = 0; i < num_snp; i++){
      	pFreq = 0;
      	for(j = 0; j < num_id; j++){    		
	      	pFreq +=  marker_mat[i*num_id+j]/(2*num_id);			   
        }
        for(j = 0; j < num_id; j++){      	
        	marker_mat[i*num_id+j] -= 2*pFreq; 	
        }
        //scale += 2*pFreq*(1-pFreq);
    }
    
    FILE *out_res = fopen(out_file, "w");
	if(out_res==NULL){
		printf("Fail to build the output file.\n");
		exit(1);
	}
	fprintf(out_res, "%s %s %s\n", "snp_0", "snp_1", "eff");
	
	char bar[102];
	for(i=0; i<=100; i++){
		bar[i] = '#';
	}
	#pragma omp parallel for schedule(guided, 5)
	for(i = 0; i < len_snp_lst_0; i++){
		//printf("%ld %ld ", i, snp_lst_0[i]);
		print_outAA(i, snp_lst_0, num_snp, num_id, marker_mat, pymat, eff_cut, out_res, len_snp_lst_0, bar);
	}
	printf("\r[%-101s] [%3d%%]\n", bar, 100);
	fflush(stdout);
	fclose(out_res); 
	out_res = NULL; 
	free(marker_mat) ;
	marker_mat = NULL;
	return 1;
}

/***********加加互作maf ***********/ 

int print_outAA_maf(long long i, long long *snp_lst_0, long long num_snp, long long num_id, double *marker_mat, double *pymat, 
        long long *freq, double *eff_cut, FILE *out_res, long long len_snp_lst_0, char *bar) {
	/*输出互作效应，为openmp服务 */
	long long j=0, k=0;
	double epi_effect=0.0;
	clock_t start, finish;
	double  duration;
	start = clock();
	for(j = snp_lst_0[i]+1; j < num_snp; j++){
		epi_effect = 0.0;
	    for(k = 0; k < num_id; k++){
			epi_effect += marker_mat[snp_lst_0[i]*num_id + k] * marker_mat[j*num_id + k] * pymat[k];
		}
		if(fabs(epi_effect) > eff_cut[freq[snp_lst_0[i]]*10 + freq[j]]){
        	fprintf(out_res, "%lld %lld %g\n", snp_lst_0[i], j, epi_effect);
    	}
	}
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	int bar_len = (int) i*100.0/len_snp_lst_0 + 1;
	char bari[102];
	strncpy(bari, bar, bar_len);
	printf("\r[%-101s] [%3d%%] [consuming time: %g seconds]", bari, bar_len, duration);
	fflush(stdout);
	return 0;
}




int remma_epiAA_maf_eff_cpu(char *bed_file, long long num_id, long long num_snp, 
     long long *snp_lst_0, long long len_snp_lst_0, double *pymat, long long *freq, double *eff_cut, char* out_file)
{
	/*加加上位*/ 
	//标记矩阵声明空间
	double *marker_mat = (double*) calloc(num_id*num_snp, sizeof(double));
	read_plink_bed(bed_file, num_id, num_snp, marker_mat);
	
	//标记矩阵中心化，-2p 
	long long i = 0, j = 0;
    double pFreq = 0.0;// frequence of one allele for each SNP
    // long long *freq = (long long*) calloc(num_snp, sizeof(long long));
    //double scale = 0;// scale factor
    for(i = 0; i < num_snp; i++){
      	pFreq = 0;
      	for(j = 0; j < num_id; j++){    		
	      	pFreq +=  marker_mat[i*num_id+j]/(2*num_id);			   
        }
        for(j = 0; j < num_id; j++){      	
        	marker_mat[i*num_id+j] -= 2*pFreq; 	
        }
        // freq[i] = (long long) (pFreq*20);
        //scale += 2*pFreq*(1-pFreq);
    }
    
    FILE *out_res = fopen(out_file, "w");
	if(out_res==NULL){
		printf("Fail to build the output file.\n");
		exit(1);
	}
	fprintf(out_res, "%s %s %s\n", "snp_0", "snp_1", "eff");
	
	char bar[102];
	for(i=0; i<=100; i++){
		bar[i] = '#';
	}
	#pragma omp parallel for schedule(guided, 5)
	for(i = 0; i < len_snp_lst_0; i++){
		//printf("%ld %ld ", i, snp_lst_0[i]);
		print_outAA_maf(i, snp_lst_0, num_snp, num_id, marker_mat, pymat, freq, eff_cut, out_res, len_snp_lst_0, bar);
	}
	printf("\r[%-101s] [%3d%%]\n", bar, 100);
	fflush(stdout);
	fclose(out_res); 
	out_res = NULL; 
	free(marker_mat) ;
	marker_mat = NULL;
	return 1;
}



/***********加显互作 ***********/ 


int print_outAD(long long i, long long *snp_lst_0, long long num_snp, long long num_id, double *marker_matA, double *marker_matD, 
        double *pymat, double eff_cut, FILE *out_res, long long len_snp_lst_0, char *bar) {
	long long j=0, k=0;
	double epi_effect=0.0;
	clock_t start, finish;
	double  duration;
	start = clock();
	for(j = snp_lst_0[i]+1; j < num_snp; j++){
		epi_effect = 0.0;
	    for(k = 0; k < num_id; k++){
			epi_effect += marker_matA[snp_lst_0[i]*num_id + k] * marker_matD[j*num_id + k] * pymat[k];
		}
		if(fabs(epi_effect) >= eff_cut){
        	fprintf(out_res, "%lld %lld %g\n", snp_lst_0[i], j, epi_effect);
    	}
    	epi_effect = 0.0;
	    for(k = 0; k < num_id; k++){
			epi_effect += marker_matD[snp_lst_0[i]*num_id + k] * marker_matA[j*num_id + k] * pymat[k];
		}
		if(fabs(epi_effect) > eff_cut){
        	fprintf(out_res, "%lld %lld %g\n", j, snp_lst_0[i], epi_effect);
    	}
	}
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	int bar_len = (int) i*100.0/len_snp_lst_0 + 1;
	char bari[102];
	strncpy(bari, bar, bar_len);
	printf("\r[%-101s] [%3d%%] [consuming time: %g seconds]", bari, bar_len, duration);
	fflush(stdout);
	return 0;
}


int remma_epiAD_eff_cpu(char *bed_file, long long num_id, long long num_snp, 
     long long *snp_lst_0, long long len_snp_lst_0, double *pymat, double eff_cut, char* out_file)
{
	//标记矩阵声明空间
	double *marker_matA = (double*) calloc(num_id*num_snp, sizeof(double));
	double *marker_matD = (double*) calloc(num_id*num_snp, sizeof(double));
	read_plink_bed(bed_file, num_id, num_snp, marker_matA);
	
	//标记矩阵中心化，-2p 
	long long i = 0, j = 0;
    double pFreq = 0.0;// frequence of one allele for each SNP
    //double scale = 0;// scale factor
    for(i = 0; i < num_snp; i++){
      	pFreq = 0;
      	for(j = 0; j < num_id; j++){    		
	      	pFreq +=  marker_matA[i*num_id+j]/(2*num_id);			   
        }
        for(j = 0; j < num_id; j++){
			if(fabs(marker_matA[i*num_id+j] - 2.0) < 0.0001){
				marker_matD[i*num_id+j] = 0.0;
			}else{
				marker_matD[i*num_id+j] = marker_matA[i*num_id+j];
			}        	
        	marker_matA[i*num_id+j] -= 2*pFreq;    	
        	marker_matD[i*num_id+j] -= 2*pFreq*(1-pFreq); 	
        }
        //scale += 2*pFreq*(1-pFreq);
    }
    
    FILE *out_res = fopen(out_file, "w");
	if(out_res==NULL){
		printf("Fail to build the output file.\n");
		exit(1);
	}
	fprintf(out_res, "%s %s %s\n", "snp_0", "snp_1", "eff");
	
	char bar[102];
	for(i=0; i<=100; i++){
		bar[i] = '#';
	}
	#pragma omp parallel for schedule(guided, 5)
	for(i = 0; i < len_snp_lst_0; i++){
		//printf("%ld %ld ", i, snp_lst_0[i]);
		print_outAD(i, snp_lst_0, num_snp, num_id, marker_matA, marker_matD, pymat, eff_cut, out_res, len_snp_lst_0, bar);
	}
	printf("\r[%-101s] [%3d%%]\n", bar, 100);
	fflush(stdout);
	fclose(out_res);  
	out_res = NULL; 
	free(marker_matA) ;
	marker_matA = NULL;
	free(marker_matD) ;
	marker_matD = NULL;
	return 1;
}


/***********加显互作maf ***********/ 


int print_outAD_maf(long long i, long long *snp_lst_0, long long num_snp, long long num_id, double *marker_matA, double *marker_matD, 
        double *pymat, long long *freqA, long long *freqD, double *eff_cut, FILE *out_res, long long len_snp_lst_0, char *bar) {
	long long j=0, k=0;
	double epi_effect=0.0;
	clock_t start, finish;
	double  duration;
	start = clock();
	for(j = snp_lst_0[i]+1; j < num_snp; j++){
		epi_effect = 0.0;
	    for(k = 0; k < num_id; k++){
			epi_effect += marker_matA[snp_lst_0[i]*num_id + k] * marker_matD[j*num_id + k] * pymat[k];
		}
		if(fabs(epi_effect) >= eff_cut[freqA[snp_lst_0[i]]*10 + freqD[j]]){
        	fprintf(out_res, "%lld %lld %g\n", snp_lst_0[i], j, epi_effect);
    	}
    	epi_effect = 0.0;
	    for(k = 0; k < num_id; k++){
			epi_effect += marker_matD[snp_lst_0[i]*num_id + k] * marker_matA[j*num_id + k] * pymat[k];
		}
		if(fabs(epi_effect) > eff_cut[freqA[snp_lst_0[i]]*10 + freqD[j]]){
        	fprintf(out_res, "%lld %lld %g\n", j, snp_lst_0[i], epi_effect);
    	}
	}
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	int bar_len = (int) i*100.0/len_snp_lst_0 + 1;
	char bari[102];
	strncpy(bari, bar, bar_len);
	printf("\r[%-101s] [%3d%%] [consuming time: %g seconds]", bari, bar_len, duration);
	fflush(stdout);
	return 0;
}


int remma_epiAD_maf_eff_cpu(char *bed_file, long long num_id, long long num_snp, 
     long long *snp_lst_0, long long len_snp_lst_0, double *pymat, 
	 long long *freqA, long long *freqD, double *eff_cut, char* out_file)
{
	//标记矩阵声明空间
	double *marker_matA = (double*) calloc(num_id*num_snp, sizeof(double));
	double *marker_matD = (double*) calloc(num_id*num_snp, sizeof(double));
	read_plink_bed(bed_file, num_id, num_snp, marker_matA);
	
	//标记矩阵中心化，-2p 
	long long i = 0, j = 0;
    double pFreq = 0.0;// frequence of one allele for each SNP
    //double scale = 0;// scale factor
    for(i = 0; i < num_snp; i++){
      	pFreq = 0;
      	for(j = 0; j < num_id; j++){    		
	      	pFreq +=  marker_matA[i*num_id+j]/(2*num_id);			   
        }
        for(j = 0; j < num_id; j++){
			if(fabs(marker_matA[i*num_id+j] - 2.0) < 0.0001){
				marker_matD[i*num_id+j] = 0.0;
			}else{
				marker_matD[i*num_id+j] = marker_matA[i*num_id+j];
			}        	
        	marker_matA[i*num_id+j] -= 2*pFreq;    	
        	marker_matD[i*num_id+j] -= 2*pFreq*(1-pFreq); 	
        }
        //scale += 2*pFreq*(1-pFreq);
    }
    
    FILE *out_res = fopen(out_file, "w");
	if(out_res==NULL){
		printf("Fail to build the output file.\n");
		exit(1);
	}
	fprintf(out_res, "%s %s %s\n", "snp_0", "snp_1", "eff");
	
	char bar[102];
	for(i=0; i<=100; i++){
		bar[i] = '#';
	}
	#pragma omp parallel for schedule(guided, 5)
	for(i = 0; i < len_snp_lst_0; i++){
		//printf("%ld %ld ", i, snp_lst_0[i]);
		print_outAD_maf(i, snp_lst_0, num_snp, num_id, marker_matA, marker_matD, pymat, freqA, freqD, eff_cut, out_res, len_snp_lst_0, bar);
	}
	printf("\r[%-101s] [%3d%%]\n", bar, 100);
	fflush(stdout);
	fclose(out_res);  
	out_res = NULL; 
	free(marker_matA) ;
	marker_matA = NULL;
	free(marker_matD) ;
	marker_matD = NULL;
	return 1;
}


/***********显显互作 ***********/ 


int print_outDD(long long i, long long *snp_lst_0, long long num_snp, long long num_id, double *marker_mat, double *pymat, 
        double eff_cut, FILE *out_res, long long len_snp_lst_0, char *bar) {
	long long j=0, k=0;
	double epi_effect=0.0;
	clock_t start, finish;
	double duration;
	start = clock();
	for(j = snp_lst_0[i]+1; j < num_snp; j++){
		epi_effect = 0.0;
	    for(k = 0; k < num_id; k++){
			epi_effect += marker_mat[snp_lst_0[i]*num_id + k] * marker_mat[j*num_id + k] * pymat[k];
		}
		if(fabs(epi_effect) > eff_cut){
        	fprintf(out_res, "%lld %lld %g\n", snp_lst_0[i], j, epi_effect);
    	}
	}
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	int bar_len = (int) i*100.0/len_snp_lst_0 + 1;
	char bari[102];
	strncpy(bari, bar, bar_len);
	printf("\r[%-101s] [%3d%%] [consuming time: %g seconds]", bari, bar_len, duration);
	fflush(stdout);
	return 0;
}



int remma_epiDD_eff_cpu(char *bed_file, long long num_id, long long num_snp, 
     long long *snp_lst_0, long long len_snp_lst_0, double *pymat, double eff_cut, char* out_file)
{
	//标记矩阵声明空间
	double *marker_mat = (double*) calloc(num_id*num_snp, sizeof(double));
	read_plink_bed(bed_file, num_id, num_snp, marker_mat);
	
	//标记矩阵中心化，-2pq 
	long long i = 0, j = 0;
    double pFreq = 0.0;// frequence of one allele for each SNP
    //double scale = 0;// scale factor
    for(i = 0; i < num_snp; i++){
      	pFreq = 0;
      	for(j = 0; j < num_id; j++){    		
	      	pFreq +=  marker_mat[i*num_id+j]/(2*num_id);			   
        }
        for(j = 0; j < num_id; j++){
			if(fabs(marker_mat[i*num_id+j] - 2.0) < 0.0001){
				marker_mat[i*num_id+j] = 0.0;
			}      	
        	marker_mat[i*num_id+j] -= 2*pFreq*(1-pFreq); 	
        }
        //scale += 2*pFreq*(1-pFreq);
    }
    
    FILE *out_res = fopen(out_file, "w");
	if(out_res==NULL){
		printf("Fail to build the output file.\n");
		exit(1);
	}
	fprintf(out_res, "%s %s %s\n", "snp_0", "snp_1", "eff");
	
	char bar[102];
	for(i=0; i<=100; i++){
		bar[i] = '#';
	}
	#pragma omp parallel for schedule(guided, 5)
	for(i = 0; i < len_snp_lst_0; i++){
		//printf("%ld %ld ", i, snp_lst_0[i]);
		print_outDD(i, snp_lst_0, num_snp, num_id, marker_mat, pymat, eff_cut, out_res, len_snp_lst_0, bar);
	}
	printf("\r[%-101s] [%3d%%]\n", bar, 100);
	fflush(stdout);
	fclose(out_res); 
	out_res = NULL; 
	free(marker_mat) ;
	marker_mat = NULL; 
	return 1;
}



/***********显显互作 maf ***********/ 


int print_outDD_maf(long long i, long long *snp_lst_0, long long num_snp, long long num_id, double *marker_mat, double *pymat, 
        long long *freq, double *eff_cut, FILE *out_res, long long len_snp_lst_0, char *bar) {
	long long j=0, k=0;
	double epi_effect=0.0;
	clock_t start, finish;
	double duration;
	start = clock();
	for(j = snp_lst_0[i]+1; j < num_snp; j++){
		epi_effect = 0.0;
	    for(k = 0; k < num_id; k++){
			epi_effect += marker_mat[snp_lst_0[i]*num_id + k] * marker_mat[j*num_id + k] * pymat[k];
		}
		if(fabs(epi_effect) > eff_cut[freq[snp_lst_0[i]]*10 + freq[j]]){
        	fprintf(out_res, "%lld %lld %g\n", snp_lst_0[i], j, epi_effect);
    	}
	}
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	int bar_len = (int) i*100.0/len_snp_lst_0 + 1;
	char bari[102];
	strncpy(bari, bar, bar_len);
	printf("\r[%-101s] [%3d%%] [consuming time: %g seconds]", bari, bar_len, duration);
	fflush(stdout);
	return 0;
}



int remma_epiDD_maf_eff_cpu(char *bed_file, long long num_id, long long num_snp, 
     long long *snp_lst_0, long long len_snp_lst_0, double *pymat, long long *freq, double *eff_cut, char* out_file)
{
	//标记矩阵声明空间
	double *marker_mat = (double*) calloc(num_id*num_snp, sizeof(double));
	read_plink_bed(bed_file, num_id, num_snp, marker_mat);
	
	//标记矩阵中心化，-2pq 
	long long i = 0, j = 0;
    double pFreq = 0.0;// frequence of one allele for each SNP
    //double scale = 0;// scale factor
    for(i = 0; i < num_snp; i++){
      	pFreq = 0;
      	for(j = 0; j < num_id; j++){    		
	      	pFreq +=  marker_mat[i*num_id+j]/(2*num_id);			   
        }
        for(j = 0; j < num_id; j++){
			if(fabs(marker_mat[i*num_id+j] - 2.0) < 0.0001){
				marker_mat[i*num_id+j] = 0.0;
			}      	
        	marker_mat[i*num_id+j] -= 2*pFreq*(1-pFreq); 	
        }
        //scale += 2*pFreq*(1-pFreq);
    }
    
    FILE *out_res = fopen(out_file, "w");
	if(out_res==NULL){
		printf("Fail to build the output file.\n");
		exit(1);
	}
	fprintf(out_res, "%s %s %s\n", "snp_0", "snp_1", "eff");
	
	char bar[102];
	for(i=0; i<=100; i++){
		bar[i] = '#';
	}
	#pragma omp parallel for schedule(guided, 5)
	for(i = 0; i < len_snp_lst_0; i++){
		//printf("%ld %ld ", i, snp_lst_0[i]);
		print_outDD_maf(i, snp_lst_0, num_snp, num_id, marker_mat, pymat, freq, eff_cut, out_res, len_snp_lst_0, bar);
	}
	printf("\r[%-101s] [%3d%%]\n", bar, 100);
	fflush(stdout);
	fclose(out_res); 
	out_res = NULL; 
	free(marker_mat) ;
	marker_mat = NULL; 
	return 1;
}


#include <iostream>
#include <cstring>
using namespace std;

int read_plink_bed(char *bed_file, long long num_id, long long num_snp, double *marker_mat)
{
	
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



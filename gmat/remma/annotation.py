import re
import sys


def gtf_to_gene_info(gtf_file):
    fout = open(gtf_file + '.gene_info', 'w')
    fin = open(gtf_file)
    for line in fin:
        if '#' in line:
            continue
        arr = line.split()
        if arr[2] == 'gene':
            searchObj = re.search('gene_id\s+"(.+?)".+gene_name\s+"(.+?)"', line, re.I)
            gene_id = searchObj.group(1)
            gene_name = searchObj.group(2)
            gene_info = ' '.join([arr[0], arr[3], arr[4], arr[6], gene_id, gene_name])
            fout.write(gene_info + '\n')
    fin.close()
    fout.close()


def annotation_snp_pos(res_file, bed_file, p_cut=1.0e-6):
    """
    注释互作检验的SNP信息
    :param res_file: 互作检验结果文件，前两列是互作SNP对的顺序编号，最后一列是P值
    :param bed_file: plink文件
    :param p_cut: P阈值
    :return:
    """
    snp_info = {}
    order = -1
    fin = open(bed_file + '.bim')
    for line in fin:
        order += 1
        arr = line.split()
        snp_info[str(order)] = ' '.join(arr)
    fin.close()
    fout = open(res_file + '.anno', 'w')
    fin = open(res_file)
    line = fin.readline()
    arr = line.split()
    fout.write(' '.join([arr[0], 'snp0_chr', 'snp0_ID', 'snp0_cm', 'snp0_bp', 'snp0_allele1', 'snp0_allele2',
                         arr[1], 'snp1_chr', 'snp1_ID', 'snp1_cm', 'snp1_bp', 'snp1_allele1', 'snp1_allele2']))
    fout.write(' ')
    fout.write(' '.join(arr[2:]))
    fout.write('\n')
    for line in fin:
        arr = line.split()
        if float(arr[-1]) < p_cut:
            fout.write(' '.join([arr[0], snp_info[arr[0]], arr[1], snp_info[arr[1]]]))
            fout.write(' ')
            fout.write(' '.join(arr[2:]))
            fout.write('\n')
    fin.close()
    fout.close()
    return 0


def annotation_snp_nearest_gene(bed_file, gene_file, max_distance=150000):
    """
    寻找SNP临近基因
    :param bed_file: plink文件
    :param gene_file: 基因信息文件，前三列是染色体号、起始位置、终止位置
    :param max_distance: SNP距离基因的最大距离
    :return:
    """
    gene_info = {}
    fin = open(gene_file)
    for line in fin:
        arr = line.split()
        try:
            gene_info[arr[0]].append(arr[:])
        except Exception as e:
            del e
            gene_info[arr[0]] = [arr[:]]
    fin.close()
    fout = open(bed_file + '.nearby_genes', 'w')
    fin = open(bed_file + '.bim')
    for line in fin:
        snp_info = line.strip()
        arr = line.split()
        snp_pos = int(arr[3])
        for genei in gene_info[arr[0]]:
            start = int(genei[1])
            end = int(genei[2])
            dist1 = snp_pos - start
            dist2 = snp_pos - end
            if dist1 > 0 and dist2 < 0:
                fout.write(snp_info + ' ' + ' '.join(genei) + ' within' + '\n')
            else:
                distance = min(abs(dist1), abs(dist2))
                if distance < max_distance:
                    fout.write(snp_info + ' ' + ' '.join(genei) + ' ' + str(distance) + '\n')
    fin.close()
    fout.close()



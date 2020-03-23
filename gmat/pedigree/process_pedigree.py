

def ped_trace(id_file, full_ped_file, gen=1000000):
    """
    描述
    ----
    根据给定的个体号，在全系谱文件中追溯这些个体的祖先。

    参数
    ----
    id_file: 储存ID号的文件，文件可包含任意列数，但第一列必须是待读取的ID号，不能出现缺失（即“0”）
    full_ped_file: 全系谱文件，前三列依次是个体号、父号、母号，系谱中的缺失用“0”表示。
    gen: 默认值为1000000，对于每个个体最大追溯的世代数。

    返回值
    ------
    个体及追溯到的祖先个体总数，相应的系谱关系输出到以id_file为前缀的"*.trace"文件里

    """

    # 读取个体号文件
    ped_id = {}
    ped_dct = {}
    fin = open(id_file, 'r')
    for line in fin:
        arr = line.split()
        if arr[0] == '0':
            print("0 is not allowed for id")  # 如果出现“0”，程序强制终止
            exit()
        ped_id[arr[0]] = 1
        ped_dct[arr[0]] = arr[0] + '\t0\t0'  # 给个体预设默认父母号，防止个体号没有出现在全系谱文件中
    fin.close()

    # 追溯个体的祖先
    ped_newid = set()
    for i in range(gen):
        ped_traceid = {}
        ped_len1 = len(ped_id.keys())
        fin = open(full_ped_file, 'r')
        for line in fin:
            arr = line.split()
            if arr[0] in ped_id:
                if arr[1] != '0':
                    ped_traceid[arr[1]] = 1
                if arr[2] != '0':
                    ped_traceid[arr[2]] = 1
        fin.close()
        ped_newid = set(ped_traceid.keys()) - set(ped_id.keys())
        ped_id.update(ped_traceid)
        ped_len2 = len(ped_id.keys())
        if ped_len1 == ped_len2:
            break

    # 读取系谱
    fin = open(full_ped_file)
    for line in fin:
        arr = line.split()
        ped_dct[arr[0]] = arr[0] + '\t0\t0'
        ped_dct[arr[1]] = arr[1] + '\t0\t0'
        ped_dct[arr[2]] = arr[2] + '\t0\t0'
    fin.close()
    fin = open(full_ped_file)
    for line in fin:
        arr = line.split()
        ped_dct[arr[0]] = "\t".join(arr[0:3])
    fin.close()

    # 输出
    fout = open(id_file + '.trace', 'w')
    for ikey in ped_id.keys():
        if ikey in ped_newid:
            fout.write('\t'.join([ikey, '0', '0']) + '\n')
        else:
            fout.write(ped_dct[ikey] + '\n')
    fout.close()
    return len(ped_id.keys())


def ped_correct(ped_file):
    """
    描述
    ----
    修正系谱中可能出现的错误
    (1) 个体号同时出现在父亲列和母亲列，这时将计算个体号在父亲列和母亲列出现的频率，将个体号在出现频率低的列中删除（设为缺失）；
    (2) 个体号同时出现在自己的祖先中，将对应的祖先号设为缺失，避免系谱按出生年月排序时出现死循环。

    参数
    ----
    ped_file: 系谱文件，前三列分别是个体号、父号、母号

    返回值
    ----
    修正后的系谱字典，以个体号为键、父母号组成的列表为值
    生成三个以ped_file为前缀的文件，”*.error1”存储犯第一种错误的系谱，”*.error2”存储犯第二种错误的系谱，”*.correct”存储修正后的系谱
    
    """

    # 计算个体号在父列和母列出现的频率
    sire_count = {}
    dam_count = {}
    fin = open(ped_file, 'r')
    for line in fin:
        arr = line.split()
        sire_count[arr[1]] = sire_count.get(arr[1], 0) + 1
        dam_count[arr[2]] = dam_count.get(arr[2], 0) + 1
    fin.close()

    # 将出现频率低的个体号存入列表
    id_overlap = set(sire_count.keys()) & set(dam_count.keys())   # 重叠个体号
    id_overlap.discard('0')
    sire_del, dam_del = [], []
    for val in id_overlap:
        if sire_count[val] < dam_count[val]:
            sire_del.append(val)
        else:
            dam_del.append(val)

    # 为父列和母列设置默认系谱（即个体号、0、0）
    ped_dct = {}
    fin = open(ped_file, 'r')
    for line in fin:
        arr = line.split()
        ped_dct[arr[1]] = ['0', '0']
        ped_dct[arr[2]] = ['0', '0']
    fin.close()

    # 将频率低的ID在相应列中设为缺失，并将可能出现错误的系谱输出
    fout = open(ped_file + '.error1', 'w')
    fin = open(ped_file, 'r')
    print('#Both sire and dam')
    for line in fin:
        arr = line.split()
        if arr[1] in sire_del:
            fout.write('\t'.join(arr[0:3]) + '\n')
            arr[1] = '0'
        if arr[2] in dam_del:
            fout.write('\t'.join(arr[0:3]) + '\n')
            arr[2] = '0'
        ped_dct[arr[0]] = [arr[1], arr[2]]
    fout.close()
    fin.close()
    try:
        ped_dct.pop('0')
    except Exception as e:
        del e

    # 将个体号所有祖先存入字典
    anc = {}
    for id in list(ped_dct.keys()):
        anc[id] = {}
        if ped_dct[id][0] != '0':
            anc[id][ped_dct[id][0]] = 1
        if ped_dct[id][1] != '0':
            anc[id][ped_dct[id][1]] = 1
        d = 1000
        while d > 0:
            id_key = list(anc[id].keys())
            id_key_len1 = len(id_key)
            for i in id_key:
                if i in ped_dct:
                    anc[id][ped_dct[i][0]] = 0
                    anc[id][ped_dct[i][1]] = 0
            id_key = list(anc[id].keys())
            id_key_len2 = len(id_key)
            d = id_key_len2 - id_key_len1
        try:
            anc[id].pop('0')
        except Exception as e:
            del e
    
    # 当个体号同时出现在其祖先时，对应祖先设为缺失
    fout = open(ped_file + '.error2', 'w')
    print('#Both id and its ancestor')
    for i in list(anc.keys()):
        ikey_lst = list(anc[i].keys())
        if i in ikey_lst:  # 判断个体号同时是其祖先
            stri = i + '\t' + ped_dct[i][0] + '\t' + ped_dct[i][1] + '\n'  # 输出个体号系谱
            fout.write(stri)
            for j in ikey_lst:  # 祖先对应的系谱条目
                if ped_dct[j][0] == i:
                    stri = j + '\t' + ped_dct[j][0] + '\t' + ped_dct[j][1] + '\n'  # 输出个体号同时出现在祖先中的对应系谱
                    fout.write(stri)
                    ped_dct[j][0] = '0'  # 将错误祖先个体号设为缺失
                if ped_dct[j][1] == i:
                    stri = j + '\t' + ped_dct[j][0] + '\t' + ped_dct[j][1] + '\n'  # 输出个体号同时出现在祖先中的对应系谱
                    fout.write(stri)
                    ped_dct[j][1] = '0'  # 将错误祖先个体号设为缺失
    fout.close()
    
    # 输出修正后的系谱
    fout = open(ped_file + '.correct', 'w')
    for i in ped_dct.keys():
        stri = i + '\t' + ped_dct[i][0] + '\t' + ped_dct[i][1] + '\n'
        fout.write(stri)
    fout.close()
    return ped_dct


def ped_sort(ped_file):
    """
    描述
    ----
    将系谱按出生先后排序，即个体祖先的系谱条目要出现在个体之前
    
    参数
    ----
    ped_file: 系谱文件，前三列分别是个体号、父号、母号
    
    返回值
    ------
    0，生成以原系谱文件为前缀的排序后文件".sort"
    
    """
    
    # 为系谱中所有个体号设置默认系谱
    ped_dct = {}
    fin = open(ped_file, 'r')
    for line in fin:
        arr = line.split()
        ped_dct[arr[0]] = ['0', '0']
        ped_dct[arr[1]] = ['0', '0']
        ped_dct[arr[2]] = ['0', '0']
    fin.close()
    
    # 读取系谱
    fin = open(ped_file, 'r')
    for line in fin:
        arr = line.split()
        ped_dct[arr[0]] = [arr[1], arr[2]]
    fin.close()
    
    try:
        ped_dct.pop('0')
    except Exception as e:
        del e
    
    # 输出排序后系谱
    output = {'0': 1}  # 存储已输出的个体号
    ped_len = len(ped_dct.keys())
    fout = open(ped_file + '.sort', 'w')
    while ped_len > 0:
        for ikey in list(ped_dct.keys()):
            # 如果亲本均已输出，则输出此个体系谱，并将此个体号存入字典
            if ped_dct[ikey][0] in output and ped_dct[ikey][1] in output:
                stri = '\t'.join([ikey, ped_dct[ikey][0], ped_dct[ikey][1]])
                fout.write(stri + '\n')
                output[ikey] = 1
                ped_dct.pop(ikey)
        ped_len = len(ped_dct.keys())
    fout.close()
    return 0


def ped_recode(ped_file):
    """
    描述
    ----
    将系谱中的个体号编码成从1开始的整数
    
    参数
    ----
    ped_file: 系谱文件，前三列分别是个体号、父号、母号
    
    返回值
    ------
    若成功，返回整数0，并生成两个以ped_file为前缀的文件，"*.dct"为个体号与其编码对应文件，第一列为个体号，第二列为编码后的整数；
    "*.recode"为编码后的文件
    
    """
    
    # 为系谱第一列编码
    code_dct = {"0": 0}
    code_val = 0
    fin = open(ped_file, 'r')
    for line in fin:
        arr = line.split()
        if arr[0] not in code_dct:
            code_val += 1
            code_dct[arr[0]] = code_val
    fin.close()
    
    # 为系谱第二、三列编码，并输出编码后系谱
    fout = open(ped_file + '.recode', 'w')
    fin = open(ped_file, 'r')
    for line in fin:
        arr = line.split()
        if arr[1] not in code_dct:
            code_val += 1
            code_dct[arr[1]] = code_val
        if arr[2] not in code_dct:
            code_val += 1
            code_dct[arr[2]] = code_val
        stri = "{0:d}\t{1:d}\t{2:d}\n".format(code_dct[arr[0]], code_dct[arr[1]], code_dct[arr[2]])
        fout.write(stri)
    fin.close()
    fout.close()
    
    # 输出个体号与编码对应文件
    fout = open(ped_file + '.dct', 'w')
    code_dct.pop('0')
    for ikey in code_dct.keys():
        fout.write('{0:s}\t{1:d}\n'.format(ikey, code_dct[ikey]))
    fout.close()
    return 0


def ped_completeness(ped_file, gen=5, cut=0.8):
    """
    描述
    ----
    计算个体的系谱完整度，并筛选出系谱完整度高的个体。参考文献：MacCluer et al. Inbreeding and pedigree structure in
    Standardbred horses, The Journal of Heredity 74:394-399. 1983.
    
    参数
    ----
    ped_file: 系谱文件
    gen: 默认为5，计算个体系谱完整度往前追溯的世代数
    cut: 默认为0.8，系谱筛选阈值，系谱完整度低于阈值的个体将被删除
    
    返回值
    ------
    生成两个以ped_file为前缀的文件
    "*.pec": 个体系谱复杂度文件，仅包含高于阈值的个体信息，第一列为个体号，第二列为系谱完整度
    "*.prune": 质控后的系谱信息，包含高于阈值及其gen（用于计算系谱完整度的代数）代祖先个体的系谱信息
    
    """
    
    # 为系谱中所有个体号设置默认系谱
    ped_dct = {}
    fin = open(ped_file, 'r')
    for line in fin:
        arr = line.split()
        ped_dct[arr[0]] = ['0', '0']
        ped_dct[arr[1]] = ['0', '0']
        ped_dct[arr[2]] = ['0', '0']
    fin.close()
    
    # 读取系谱信息
    fin = open(ped_file, 'r')
    for line in fin:
        arr = line.split()
        ped_dct[arr[0]] = [arr[1], arr[2]]
    fin.close()
    
    # 计算个体系谱完整度
    ped_output = {}  # 储存将被输出的个体号
    fout = open(ped_file + '.pec', 'w')
    sire1, dam1 = [], []
    for ikey in list(ped_dct.keys()):
        anc_lst = []
        if ped_dct[ikey][0] == '0' or ped_dct[ikey][1] == '0':  # 如果个体亲本有一缺失，完整度即为0
            pec_val = 0.0
        else:
            sire1, dam1 = [ped_dct[ikey][0]], [ped_dct[ikey][1]]
            anc_lst.extend(sire1)
            anc_lst.extend(dam1)
            pec_sire, pec_dam = 0.5, 0.5  # 往前一代父系、母系完整度
            for val in range(2, gen + 1):  # 往前2-gen代
                sire2, dam2 = [], []
                for id in sire1:
                    if ped_dct[id][0] != '0':
                        pec_sire += 1.0/pow(2, val)  # pow(2, val) 为往前第val世代亲本数
                        sire2.append(ped_dct[id][0])
                    if ped_dct[id][1] != '0':
                        pec_sire += 1.0/pow(2, val)
                        sire2.append(ped_dct[id][1])
                for id in dam1:
                    if ped_dct[id][0] != '0':
                        pec_dam += 1.0/pow(2, val)
                        dam2.append(ped_dct[id][0])
                    if ped_dct[id][1] != '0':
                        pec_dam += 1.0/pow(2, val)
                        dam2.append(ped_dct[id][1])
                sire1, dam1 = list(sire2), list(dam2)
                anc_lst.extend(sire1)
                anc_lst.extend(dam1)
            pec_sire /= gen
            pec_dam /= gen
            pec_val = 4 * pec_sire * pec_dam / (pec_sire + pec_dam)
        if pec_val > cut:
            fout.write("{0:s}\t{1:f}\n".format(ikey, pec_val))
            ped_output[ikey] = list(ped_dct[ikey])
            for id in anc_lst:
                if id in sire1 or id in dam1:  # 往前第gen代父母设为0、0，否则相当于追溯了gen + 1代
                    ped_output[id] = ['0', '0']
                else:
                    ped_output[id] = list(ped_dct[id])
    fout.close()
    
    # 输出质控后的系谱
    fout = open(ped_file + '.prune', 'w')
    for id in ped_output:
        fout.write(id + '\t' + '\t'.join(ped_output[id]) + '\n')
    fout.close()
    
    return 0

import numpy as np
import warnings
from common.subset_new import subsets
from common.g2test import g2_test_dis
from common.MI_discrete import mutual_information_test
import copy
import matlab.engine
import matlab

eng = matlab.engine.start_matlab()

def myunion(list1, list2):
    return list(set(list1) | set(list2))

def MCF_Sp(data, alpha, L, k1):
    warnings.filterwarnings("ignore")
    n, p = np.shape(data)

    f = p - L
    f = int(f)
    p = int(p)

    PC = [[] for i in range(p)]
    da = data.tolist()
    for i in range(L):
        target = i + f
        target = target + 1
        pc = eng.getpc_new_g2(matlab.double(da), target, 0.05)
        target = target - 1

        if isinstance(pc, float):
            PC[target].append(int(pc))
        else:
            if len(pc) != 0:
                for j in pc[0]:
                    PC[target].append(int(j))

    min_dep = [1 for i in range(p)]
    for i in range(f, p):
        min_dep[i] = 1
        for j in PC[i]:
            pval, dep = g2_test_dis(data, i, j, [], alpha)
            if dep < min_dep[i]:
                min_dep[i] = dep

    record = []
    for i in range(f, p):
        for j in range(f, p):
            if j in PC[i] and i not in record:
                record.append(i)
    record = sorted(list(set(record)))

    fk_pc = [[] for i in range(p)]
    fea_lab = [[0 for i in range(f)] for i in range(p)]
    for j in record:
        pc_dep = []
        fk = [i for i in range(f) if i not in PC[j]]
        for k in fk:
            pval, dep = g2_test_dis(data, j, k, [], alpha)
            fea_lab[j][k] = dep
            if pval <= alpha and dep >= min_dep[j]:
                pc_dep.append([k, dep])
        pc_dep = sorted(pc_dep, key=lambda x: x[1], reverse=True)
        sellen = int(len(pc_dep) * k1)
        for k in range(sellen):
            fk_pc[j].append(pc_dep[k][0])

    pc_subset = [[] for i in range(p)]
    for j in record:
        for i in range(3):
            pc_subset_temp = subsets(PC[j], i)
            for k in pc_subset_temp:
                pc_subset[j].append(k)

    """phase 2"""
    for j in record:
        for i in range(f, p):
            if i in PC[j]:
                for fk in fk_pc[j]:
                    flag = False
                    for s in pc_subset[j]:
                        s_temp = s
                        if len(s_temp) < len(PC[j]) and i not in s_temp:
                            s_temp.append(i)
                        if i not in s_temp:
                            continue
                        pval, mi_xs = mutual_information_test(data, j, fk, s_temp)
                        if pval > alpha:
                            s_temp.remove(i)
                            vars = list(set(s_temp))
                            pval1, mi_xs = mutual_information_test(data, j, fk, vars)
                            if pval1 > alpha:
                                flag = True
                                break
                    if flag == False:
                        PC[j].append(fk)
                PC[j].remove(i)

    for i in range(f, p):
        PC[i] = list(set(PC[i]))
    for i in range(f, p):
        for j in range(f, p):
            if j in PC[i]:
                PC[i].remove(j)

    """phase 3"""
    MB = [[] for i in range(p)]
    for i in range(f, p):
        MB[i] = copy.deepcopy(PC[i])

    for j in range(f, p):
        for i in range(f):
            if i not in PC[j] and len([p_ for p_ in range(f, p) if j in PC[p_]]) > 1:
                SP = eng.getpc_new_g2(matlab.double(da), i + 1, 0.05)
                if isinstance(SP, float):
                    SP = [int(SP)]
                else:
                    SP = [int(x) for x in SP[0]]

                for x in SP:
                    if x not in PC[j]:
                        vars = PC[j] + SP + [x]
                        vars = list(set(vars) - {x})
                        pval, dep = g2_test_dis(data, j, x, vars, alpha)
                        if pval > alpha:
                            PC[j].remove(x)
                MB[j] = list(set(MB[j] + SP))

    selfea = []
    for j in range(f, p):
        for i in MB[j]:
            if i not in selfea:
                selfea.append(i)

    selfea = list(set(selfea))
    return selfea
from itertools import combinations

def subsets(nbrs,k):
    s=combinations(nbrs,k)
    sub=[]
    for i in s:
        sub.append(list(i))
    return sub
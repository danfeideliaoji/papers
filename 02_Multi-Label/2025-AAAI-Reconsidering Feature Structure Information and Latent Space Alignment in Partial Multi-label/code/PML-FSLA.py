import numpy as np
from numpy.random import seed
from numpy import linalg as LA
from scipy import linalg
from scipy.spatial import distance
from skfeature.utility.construct_W import construct_W
import skfeature.utility.entropy_estimators as ees
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import scipy.io as scio
import scipy
import pandas as pd
import math
import pandas as pd
import skfeature.utility.entropy_estimators as ees
from sklearn.metrics.pairwise import cosine_similarity
import skfeature.utility.entropy_estimators as ees
import time
from sklearn.cluster import OPTICS
from sklearn.decomposition import NMF
from sklearn.datasets import make_blobs
eps = 2.2204e-16

from scipy.optimize import minimize




def single_update_T_and_lambda(X, P, W, M, T, lambdas, beta, lr=0.01, lambda_lr=0.01):
    """
    单次更新矩阵 T 和拉格朗日乘子 lambdas。

    参数:
    - X, P, W, M: 输入矩阵。
    - T: 当前的矩阵 T。
    - lambdas: 当前的拉格朗日乘子。
    - beta: 正则化参数。
    - lr: 学习率用于更新 T。
    - lambda_lr: 拉格朗日乘子的更新率。

    返回:
    - 更新后的矩阵 T 和拉格朗日乘子。
    """
    m, n = T.shape
    
    # 计算拉格朗日函数关于 T 的梯度
    grad_T = 2 * (T - X @ W) + 2 * beta * (T - P @ M) + lambdas[:, np.newaxis]

    # 更新 T
    T = T - lr * grad_T

    # 保证 T 的每行和为1
    row_sums = T.sum(axis=1)
    T = T / row_sums[:, np.newaxis]

    # 更新拉格朗日乘子
    lambdas = lambdas + lambda_lr * (1 - row_sums)

    return T, lambdas

# 示例使用
# 假设 X, P, W, M, T, 和 lambdas 都是已定义的 NumPy 数组
# 初始化拉格朗日乘子
# lambdas = np.zeros(T.shape[0])
# 在一个循环中多次调用这个函数以逐步更新 T 和 lambdas


# 示例使用
# 假设 X, P, W, M, T, 和 lambdas 都是已定义的 NumPy 数组
# 初始化拉格朗日乘子
# lambdas = np.zeros(T.shape[0])
# 在一个循环中多次调用这个函数以逐步更新 T 和 lambdas



def auto_cluster(X):
    optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.01)
    labels = optics.fit_predict(X)

    # 获取聚类的数量（排除噪声点，假设噪声标签为-1）
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # 构造聚类矩阵U
    U = np.zeros((X.shape[0], n_clusters), dtype=np.float32)
    for i, label in enumerate(labels):
        if label != -1:  # 排除噪声点
            U[i, label] = 1

    # 计算每个簇的特征平均值来构造V
    V = np.zeros((n_clusters, X.shape[1]))
    for label in range(n_clusters):
        V[label, :] = X[labels == label].mean(axis=0)
    return U,V,n_clusters
    # 显示结果
    print("矩阵U (聚类矩阵):\n", U)
    print("矩阵V:\n", V)
def k_cluster(X,k):
    model = NMF(n_components=k, init='random', random_state=0)

    # 拟合数据并进行转换
    U = model.fit_transform(X)
    V = model.components_

    return U,V

def my_pml(X, Y,para1,para2,para3,para4):
    time_start = time.time()
    n,d=X.shape
    n,l=Y.shape
    U,V,k=auto_cluster(X)
    U=np.random.rand(n, k)
    V=np.random.rand(k, d)
    W = np.random.rand(d, l)
    #P,R=k_cluster(Y,k)
    P= np.random.rand(n, k)
    R= np.random.rand(k, l)
    options = {'metric': 'euclidean', 'neighbor_mode': 'knn', 'k': 5, 'weight_mode': 'heat_kernel', 't': 1.0}
    Tr=Y
    lambdas = np.zeros(n)
    #print(U.shape,V.shape,R.shape,P.shape,W.shape)
    t=0
    obji = 1
    fun_ite=[]
    cver=0
    while 1:
        S = construct_W(P, **options)
        S = S.A
        A = np.diag(np.sum(S, 0))
        L = A - S
        Btmp = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        d1 = 0.5 / Btmp
        D = np.diag(d1.flat)
         
        Tr,lambdas=single_update_T_and_lambda(X, P, W, R, Tr,lambdas,para2)
        V=np.multiply(V, np.true_divide(np.dot(U.T,X) ,np.dot(np.dot(U.T,U),V) +eps))
        U=np.multiply(U, np.true_divide(para1*np.dot(X,V.T)+para3*P,para1*np.dot(np.dot(U,V),V.T) +para3*U+eps))
        R=np.multiply(R, np.true_divide(np.dot(P.T,Tr) ,np.dot(np.dot(P.T,P),R) +eps))
        P=np.multiply(P, np.true_divide(para3*U+para2*np.dot(Tr,R.T) ,para3*P+para2*np.dot(np.dot(P,R),R.T) +eps))

        #Tr=np.dot(P,R)
        W=np.multiply(W, np.true_divide(np.dot(X.T,Tr) ,np.dot(np.dot(X.T,X),W) +para4*np.dot(D,W)+eps))
        fun= pow(LA.norm(np.dot(X, W)-Tr, 'fro'), 2)+para1*pow(LA.norm(X-np.dot(U, V), 'fro'), 2)+para2*pow(LA.norm(Tr-np.dot(P, R), 'fro'), 2)+para3*pow(LA.norm(U-P, 'fro'), 2)+para4*np.trace(np.dot(np.dot(W.T, D), W))
        fun_ite.append(fun)
        cver = abs((fun - obji) / float(obji))
        obji = fun
        t=t+1
        print(fun,t)
        if fun<100:
            if (t > 2 and (cver < 0.1 or t == 1000)):
                break
        else:
            if (t > 2 and (cver < 1e-3 or t == 1000)):
                #print (Tr,Y)
                #W=np.dot(W,S)
                break
    time_end = time.time()
    running_time = time_end - time_start
    score = np.sum(np.multiply(W, W), 1)
    idx = np.argsort(-score, axis=0)
    idx = idx.tolist()
    l = [i for i in idx]

    return l, W

def my_pml1(X, Y,para1,para2,para3):
    time_start = time.time()
    n,d=X.shape
    n,l=Y.shape
    L,Qt,k=auto_cluster(X)
    Q=Qt.T
    Q=np.random.rand(d, k)
    L=np.random.rand(n, k)
    #P,M=k_cluster(Y,k)
    P=np.random.rand(n, k)
    M=np.random.rand(k, l)
    options = {'metric': 'euclidean', 'neighbor_mode': 'knn', 'k': 5, 'weight_mode': 'heat_kernel', 't': 1.0}
    Tr=Y
    #print(U.shape,V.shape,R.shape,P.shape,W.shape)
    t=0
    obji = 1
    fun_ite=[]
    cver=0
    while 1:
        tem=np.dot(Q,M)
        Btmp = np.sqrt(np.sum(np.multiply(tem, tem), 1) + eps)
        d1 = 0.5 / Btmp
        D = np.diag(d1.flat)
        Q=np.multiply(Q, np.true_divide(np.dot(X.T,L) ,np.dot(np.dot(Q,L.T),L) +para3*np.dot(np.dot(D,tem),M.T)+eps))
        L=np.multiply(L, np.true_divide(np.dot(X,Q)+para2*P ,np.dot(np.dot(L,Q.T),Q) +para2*L+eps))
        Tr=np.multiply(Tr, np.true_divide(np.dot(P,M) ,Tr+eps))
        row_sums = Tr.sum(axis=1)
        Tr /= row_sums[:, np.newaxis]
        P=np.multiply(P, np.true_divide(para1*np.dot(Tr,M.T)+para2*L ,np.dot(np.dot(P,M),M.T) +para2*P+eps))
        M=np.multiply(M, np.true_divide(np.dot(P.T,Tr) ,np.dot(np.dot(P.T,P),M)+para3*np.dot(np.dot(Q.T,D),tem)+eps))
        fun= pow(LA.norm(X-np.dot(L, Q.T), 'fro'), 2)+para1*pow(LA.norm(Tr-np.dot(P, M), 'fro'), 2)+para2*pow(LA.norm(L-P, 'fro'), 2)+para3*np.trace(np.dot(np.dot(tem.T, D),tem ))
        fun_ite.append(fun)
        cver = abs((fun - obji) / float(obji))
        obji = fun
        t=t+1
        print(fun,t)
        if fun<100:
            if (t > 2 and (cver < 0.1 or t == 1000)):
                break
        else:
            if (t > 2 and (cver < 1e-3 or t == 1000)):
                #print (Tr,Y)
                #W=np.dot(W,S)
                break
        
    time_end = time.time()
    running_time = time_end - time_start
    score = np.sum(np.multiply(Q, Q), 1)
    idx = np.argsort(-score, axis=0)
    idx = idx.tolist()
    l = [i for i in idx]
    W=np.dot(Q,M)
    return l, W,fun_ite

def my_pml0(X, Y,para1,para2,para3):
    time_start = time.time()
    n,d=X.shape
    n,l=Y.shape
    a=l
    L,Qt,k=auto_cluster(X)
    Q=Qt.T
    Q=np.random.rand(d, k)
    L=np.random.rand(n, k)
    #P,M=k_cluster(Y,k)
    P=np.random.rand(n, k)
    M=np.random.rand(k, l)
    options = {'metric': 'euclidean', 'neighbor_mode': 'knn', 'k': 5, 'weight_mode': 'heat_kernel', 't': 1.0}
    Tr=Y
    #print(U.shape,V.shape,R.shape,P.shape,W.shape)
    t=0
    obji = 1
    fun_ite=[]
    cver=0
    while 1:
        tem=np.dot(Q,M)
        Btmp = np.sqrt(np.sum(np.multiply(tem, tem), 1) + eps)
        d1 = 0.5 / Btmp
        D = np.diag(d1.flat)
        Q=np.multiply(Q, np.true_divide(np.dot(X.T,L) ,np.dot(np.dot(Q,L.T),L) +para3*np.dot(np.dot(D,tem),M.T)+eps))
        L=np.multiply(L, np.true_divide(np.dot(X,Q)+para2*P ,np.dot(np.dot(L,Q.T),Q) +para2*L+eps))
        Tr=np.multiply(Tr, np.true_divide(np.dot(P,M) ,Tr+eps))
        row_sums = Tr.sum(axis=1)
        Tr /= row_sums[:, np.newaxis]
        P=np.multiply(P, np.true_divide(para1*np.dot(Tr,M.T)+para2*L ,np.dot(np.dot(P,M),M.T) +para2*P+eps))
        M=np.multiply(M, np.true_divide(np.dot(P.T,Tr) ,np.dot(np.dot(P.T,P),M)+para3*np.dot(np.dot(Q.T,D),tem)+eps))
        fun= pow(LA.norm(X-np.dot(L, Q.T), 'fro'), 2)+para1*pow(LA.norm(Tr-np.dot(P, M), 'fro'), 2)+para2*pow(LA.norm(L-P, 'fro'), 2)+para3*np.trace(np.dot(np.dot(tem.T, D),tem ))
        fun_ite.append(fun)
        cver = abs((fun - obji) / float(obji))
        obji = fun
        t=t+1
        print(fun,t)
        if fun<100:
            if (t > 2 and (cver < 0.1 or t == 1000)):
                break
        else:
            if (t > 2 and (cver < 1e-3 or t == 1000)):
                #print (Tr,Y)
                #W=np.dot(W,S)
                break
        
    time_end = time.time()
    running_time = time_end - time_start
    score = np.sum(np.multiply(Q, Q), 1)
    idx = np.argsort(-score, axis=0)
    idx = idx.tolist()
    l = [i for i in idx]
    bu=np.zeros((d,a-k))
    W=np.concatenate((Q, bu), axis=1)
    return l, W
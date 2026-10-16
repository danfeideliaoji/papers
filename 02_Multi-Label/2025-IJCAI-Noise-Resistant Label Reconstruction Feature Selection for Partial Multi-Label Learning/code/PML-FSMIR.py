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
eps = 2.2204e-16
def getD(X,Y):
    n,d=X.shape
    n,l=Y.shape
    Dlist=np.zeros((d,l))
    for i in range(d):
        for j in range(l):
            Dlist[i,j]=ees.midd(X[:,i],Y[:,j])
    return Dlist
def getMI(X):
    n,d=X.shape
    MI=np.zeros((d,d))
    for i in range(d):
        for j in range(i,d):
            MI[i,j]=ees.midd(X[:,i],X[:,j])
            MI[j,i]=ees.midd(X[:,i],X[:,j])
    return MI
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def dizhi(X, Y,para1,para2,para3):
    time_start = time.time()
    n,d=X.shape
    n,l=Y.shape
    
    Z=getMI(Y)
    print(n,d,l)
    k=int(np.sqrt(d))
    U = np.random.rand(n, k)
    V = np.random.rand(k, d)
    W = np.random.rand(d, l)
    #Ws=getD(X,Y)
    Xs=X


    #Y=np.multiply(Y,np.dot(Y,Z))
    #Y=normalization(Y)
    #Ws=getD(X,Y)
    
    S=getMI(Y)
    #S1=getMI(X)
    A = np.diag(np.sum(S, 0))
    L = A - S
    t=0
    obji = 1
    fun_ite=[]
    cver=0
    while 1:
        
        Btmp = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        d1 = 0.5 / Btmp
        D = np.diag(d1.flat)
        
        U=np.multiply(U, np.true_divide(np.dot(Xs,V.T) + para1*np.dot(np.dot(Y,W.T),V.T), np.dot(np.dot(U,V),V.T)+ para1*np.dot(np.dot(np.dot(np.dot(U,V),W),W.T),V.T)+eps))
        V=np.multiply(V, np.true_divide(np.dot(U.T,Xs) + para1*np.dot(np.dot(U.T,Y),W.T), np.dot(np.dot(U.T,U),V)+ para1*np.dot(np.dot(np.dot(np.dot(U.T,U),V),W),W.T)+ eps))
        W=np.multiply(W, np.true_divide(para1*np.dot(np.dot(V.T,U.T),Y)+para2*np.dot(W,S), para1*np.dot(np.dot(np.dot(np.dot(V.T,U.T),U),V),W)+ para2*np.dot(W,A)+para3*np.dot(D,W)+eps))
        #W=np.multiply(W,Ws)
        
        
        fun= pow(LA.norm(np.dot(np.dot(U, V),W)-Y, 'fro'), 2)+para1*pow(LA.norm(Xs-np.dot(U, V), 'fro'), 2)+para2*np.trace(np.dot(np.dot(W,L),W.T))+para3*np.trace(np.dot(np.dot(W.T, D), W))
        fun_ite.append(fun)
        cver = abs((fun - obji) / float(obji))
        obji = fun
        t=t+1
        #print(fun,t)
        if (t > 2 and (cver < 1e-3 or t == 1000)):
            #W=np.dot(W,S)
            break
    time_end = time.time()
    running_time = time_end - time_start
    score = np.sum(np.multiply(W, W), 1)
    idx = np.argsort(-score, axis=0)
    idx = idx.tolist()
    l = [i for i in idx]

    return l, W,fun_ite
def dizhi1(X, Y,para1,para2):
    time_start = time.time()
    n,d=X.shape
    n,l=Y.shape
    
    Z=getMI(Y)
    print(n,d,l)
    k=int(np.sqrt(d))
    W = np.random.rand(d, l)
    Ws=getD(X,Y)
    Xs=X

    
    #Y=np.multiply(Y,np.dot(Y,Z))
    #Y=normalization(Y)
    Ws=getD(X,Y)
    
    S=getMI(Y)
    A = np.diag(np.sum(S, 0))
    L = A - S
    t=0
    obji = 1
    fun_ite=[]
    cver=0
    while 1:
        Btmp = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        d1 = 0.5 / Btmp
        D = np.diag(d1.flat)

        W=np.multiply(W, np.true_divide(np.dot(X.T,Y)+para1*np.dot(W,S), np.dot(np.dot(X.T,X),W)+ para1*np.dot(W,A)+para2*np.dot(D,W)+eps))
        #W=np.multiply(W,Ws)
        
        
        fun= pow(LA.norm(np.dot(X,W)-Y, 'fro'), 2)+para1*np.trace(np.dot(np.dot(W,L),W.T))+para2*np.trace(np.dot(np.dot(W.T, D), W))
        fun_ite.append(fun)
        cver = abs((fun - obji) / float(obji))
        obji = fun
        t=t+1
        #print(fun,t)
        if (t > 2 and (cver < 1e-3 or t == 1000)):
            #W=np.multiply(W,Ws)
            break
    time_end = time.time()
    running_time = time_end - time_start
    score = np.sum(np.multiply(W, W), 1)
    idx = np.argsort(-score, axis=0)
    idx = idx.tolist()
    l = [i for i in idx]

    return l, W,fun_ite
def dizhi2(X, Y,para1,para2,para3):
    time_start = time.time()
    n,d=X.shape
    n,l=Y.shape
    
    Z=getMI(Y)
    print(n,d,l)
    k=int(np.sqrt(d))
    U = np.random.rand(n, k)
    V = np.random.rand(k, d)
    W = np.random.rand(d, l)
    Ws=getD(X,Y)
    Xs=X


    Y=np.multiply(Y,np.dot(Y,Z))
    Y=normalization(Y)
    Ws=getD(X,Y)
    
    S=getMI(Y)
    S1=getMI(X)
    A = np.diag(np.sum(S, 0))
    L = A - S
    t=0
    obji = 1
    fun_ite=[]
    cver=0
    while 1:
        
        Btmp = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        d1 = 0.5 / Btmp
        D = np.diag(d1.flat)
        
        U=np.multiply(U, np.true_divide(np.dot(Xs,V.T) + para1*np.dot(np.dot(Y,W.T),V.T), np.dot(np.dot(U,V),V.T)+ para1*np.dot(np.dot(np.dot(np.dot(U,V),W),W.T),V.T)+eps))
        V=np.multiply(V, np.true_divide(np.dot(U.T,Xs) + para1*np.dot(np.dot(U.T,Y),W.T), np.dot(np.dot(U.T,U),V)+ para1*np.dot(np.dot(np.dot(np.dot(U.T,U),V),W),W.T)+ eps))
        W=np.multiply(W, np.true_divide(para1*np.dot(np.dot(V.T,U.T),Y)+para2*np.dot(W,S), para1*np.dot(np.dot(np.dot(np.dot(V.T,U.T),U),V),W)+ para2*np.dot(W,A)+para3*np.dot(D,W)+eps))
        #W=np.multiply(W,Ws)
        
        
        fun= pow(LA.norm(np.dot(np.dot(np.dot(U, V),S1),W)-Y, 'fro'), 2)+para1*pow(LA.norm(Xs-np.dot(U, V), 'fro'), 2)+para2*np.trace(np.dot(np.dot(W,L),W.T))+para3*np.trace(np.dot(np.dot(W.T, D), W))
        fun_ite.append(fun)
        cver = abs((fun - obji) / float(obji))
        obji = fun
        t=t+1
        #print(fun,t)
        if (t > 2 and (cver < 1e-3 or t == 1000)):
            #W=np.multiply(W,Ws)
            break
    time_end = time.time()
    running_time = time_end - time_start
    score = np.sum(np.multiply(W, W), 1)
    idx = np.argsort(-score, axis=0)
    idx = idx.tolist()
    l = [i for i in idx]

    return l, W
def dizhiY(X, Y,para1,para2):
    time_start = time.time()
    n,d=X.shape
    n,l=Y.shape
    
    W = np.random.rand(d, l)
    Z=getMI(Y)
    
    
    Y=np.multiply(Y,np.dot(Y,Z))
    Y=normalization(Y)   
    S=getMI(Y)
    A = np.diag(np.sum(S, 0))
    L = A - S
    t=0
    obji = 1
    fun_ite=[]
    cver=0
    while 1:
        Btmp1 = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        d1 = 0.5 / Btmp1
        D1 = np.diag(d1.flat)
        temp=np.dot(X,W)-Y
        Btmp2 = np.sqrt(np.sum(np.multiply(temp, temp), 1) + eps)
        d2 = 0.5 / Btmp2
        D2 = np.diag(d2.flat)
        '''
        print("X",X.shape)
        print("S",S.shape)
        print("W",W.shape)
        print("D2",D2.shape)
        print("Y",Y.shape)
        print("A",A.shape)
        print("D1",D1.shape)
        '''
        fun= np.trace(np.dot(np.dot(temp.T, D2), temp))+para1*np.trace(np.dot(np.dot(W,L),W.T))+para2*np.trace(np.dot(np.dot(W.T, D1), W))
        W=np.multiply(W, np.true_divide(para1*np.dot(W,S)+np.dot(np.dot(X.T,D2),Y), np.dot(np.dot(np.dot(X.T,D2),X),W)+para1*np.dot(W,A)+ para2*np.dot(D1,W)+eps))
        
        fun_ite.append(fun)
        cver = abs((fun - obji) / float(obji))
        obji = fun
        t=t+1
        #print(fun,t)
        if (t > 2 and (cver < 1e-3 or t == 1000)):
            print(W)
            break
    time_end = time.time()
    running_time = time_end - time_start
    
    score = np.sum(np.multiply(W, W), 1)
    idx = np.argsort(-score, axis=0)
    idx = idx.tolist()
    l = [i for i in idx]

    return l, W
def dizhiX(X, Y,para1,para2):
    time_start = time.time()
    n,d=X.shape
    n,l=Y.shape
    
    W = np.random.rand(d, l)
    Z=getMI(Y)
    
    
    Y=np.multiply(Y,np.dot(Y,Z))
    Y=normalization(Y)   
    S=getMI(X)
    A = np.diag(np.sum(S, 0))
    L = A - S
    t=0
    obji = 1
    fun_ite=[]
    cver=0
    while 1:
        Btmp1 = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        d1 = 0.5 / Btmp1
        D1 = np.diag(d1.flat)
        temp=np.dot(X,W)-Y
        Btmp2 = np.sqrt(np.sum(np.multiply(temp, temp), 1) + eps)
        d2 = 0.5 / Btmp2
        D2 = np.diag(d2.flat)
        '''
        print("X",X.shape)
        print("S",S.shape)
        print("W",W.shape)
        print("D2",D2.shape)
        print("Y",Y.shape)
        print("A",A.shape)
        print("D1",D1.shape)
        '''
        fun= np.trace(np.dot(np.dot(temp.T, D2), temp))+para1*np.trace(np.dot(np.dot(W.T,L),W))+para2*np.trace(np.dot(np.dot(W.T, D1), W))
        W=np.multiply(W, np.true_divide(para1*np.dot(S,W)+np.dot(np.dot(X.T,D2),Y), np.dot(np.dot(np.dot(X.T,D2),X),W)+para1*np.dot(A,W)+ para2*np.dot(D1,W)+eps))
        
        fun_ite.append(fun)
        cver = abs((fun - obji) / float(obji))
        obji = fun
        t=t+1
        #print(fun,t)
        if (t > 2 and (cver < 1e-3 or t == 1000)):
            print(W)
            break
    time_end = time.time()
    running_time = time_end - time_start
    score = np.sum(np.multiply(W, W), 1)
    idx = np.argsort(-score, axis=0)
    idx = idx.tolist()
    l = [i for i in idx]

    return l, W
def reverse(X,Y,para1):
    time_start = time.time()
    n,d=X.shape
    n,l=Y.shape
    #Y=1-Y
    W = np.random.rand(d, l)
    t=0
    obji = 1
    fun_ite=[]
    cver=0
    while 1:
        Btmp1 = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        d1 = 0.5 / Btmp1
        D1 = np.diag(d1.flat)
        W=np.multiply(W, np.true_divide(np.dot(X.T,Y), np.dot(np.dot(X.T,X),W)+ para1*np.dot(D1,W)+eps))
        fun=pow(LA.norm(np.dot(X,W)-Y, 'fro'), 2)+para1*np.trace(np.dot(np.dot(W.T, D1), W))
        fun_ite.append(fun)
        cver = abs((fun - obji) / float(obji))
        obji = fun
        t=t+1
        print(fun,t)
        if (t > 2 and (cver < 1000 or t == 1000)):
            #print(W)
            break
    time_end = time.time()
    running_time = time_end - time_start
    score = np.sum(np.multiply(W, W), 1)
    idx = np.argsort(-score, axis=0)
    idx = idx.tolist()
    print("wu",idx[0:int(0.2*d)])
    l = [i for i in idx]

    return l, W

def reverse1(X,Y,para1):
    time_start = time.time()
    n,d=X.shape
    n,l=Y.shape
    Y=1-Y
    W = np.random.rand(d, l)
    t=0
    obji = 1
    fun_ite=[]
    cver=0
    while 1:
        Btmp1 = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        d1 = 0.5 / Btmp1
        D1 = np.diag(d1.flat)
        W=np.multiply(W, np.true_divide(np.dot(X.T,Y), np.dot(np.dot(X.T,X),W)+ para1*np.dot(D1,W)+eps))
        fun=pow(LA.norm(np.dot(X,W)-Y, 'fro'), 2)+para1*np.trace(np.dot(np.dot(W.T, D1), W))
        fun_ite.append(fun)
        cver = abs((fun - obji) / float(obji))
        obji = fun
        t=t+1
        print(fun,t)
        if (t > 2 and (cver < 1000 or t == 1000)):
            #print(W)
            break
    time_end = time.time()
    running_time = time_end - time_start
    score = np.sum(np.multiply(W, W), 1)
    idx = np.argsort(score, axis=0)
    idx = idx.tolist()
    l = [i for i in idx]
    print("fan",idx[0:int(0.2*d)])
    return l, W

def reverse2(X,Y,para1):
    time_start = time.time()
    n,d=X.shape
    n,l=Y.shape
    Y=1-Y
    W = np.random.rand(d, l)
    t=0
    obji = 1
    fun_ite=[]
    cver=0
    while 1:
        Btmp1 = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        d1 = 0.5 / Btmp1
        D1 = np.diag(d1.flat)
        W=np.multiply(W, np.true_divide(np.dot(X.T,Y), np.dot(np.dot(X.T,X),W)+ para1*np.dot(D1,W)+eps))
        fun=pow(LA.norm(np.dot(X,W)-Y, 'fro'), 2)+para1*np.trace(np.dot(np.dot(W.T, D1), W))
        fun_ite.append(fun)
        cver = abs((fun - obji) / float(obji))
        obji = fun
        t=t+1
        print(fun,t)
        if (t > 2 and (cver < 1000 or t == 1000)):
            #print(W)
            break
    time_end = time.time()
    running_time = time_end - time_start
    score = np.sum(np.multiply(W, W), 1)
    idx = np.argsort(-score, axis=0)
    idx = idx.tolist()
    l = [i for i in idx]
    print("bufan",idx[0:int(0.2*d)])
    return l, W
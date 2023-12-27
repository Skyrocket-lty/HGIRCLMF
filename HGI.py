# -*- codeing = utf-8 -*-
# @Time : 2023/3/24 21:17
# @Author : 刘体耀
# @File : TSPN.py
# @Software: PyCharm


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy

SM = np.loadtxt(r"SM相似性矩阵-weight.txt",dtype=float)
MM = np.loadtxt(r"miRNA相似性矩阵-weight.txt",dtype=float)
Y1= np.loadtxt(r"SM-miRNA关联矩阵.txt",dtype=float)
#T = np.loadtxt(r"SM-miRNA关联矩阵-补全后.txt", dtype=int)
SM_miRNA_k = np.loadtxt(r"SM-miRNA-已知关联.txt",dtype=int)
SM_miRNA_uk = np.loadtxt(r"SM-miRNA-未知关联.txt",dtype=int)


def run_MC_3(T):
# the normalization of SM
    #print(T)
    SM1=copy.deepcopy(SM)
    for mm1 in range(831):
        for mm2 in range(831):
            SM[mm1,mm2]=SM[mm1,mm2]/(np.sqrt(np.sum(SM1[mm1,:]))*np.sqrt(np.sum(SM1[mm2,:])))

#the normalization of MM
    MM1=copy.deepcopy(MM)
    for nn1 in range(541):
        for nn2 in range(541):
            MM[nn1,nn2]=MM[nn1,nn2]/(np.sqrt(np.sum(MM1[nn1,:]))*np.sqrt(np.sum(MM1[nn2,:])))

#calculate the score matrix S
    S=np.mat(np.random.rand(831,541))
    Si=0.4*SM@S@MM+0.6*T
    while np.linalg.norm(Si-S,1)>10**-6:
        S=Si
        Si=0.4*SM@S@MM+0.6*T
    T_recovery = Si
    return T_recovery

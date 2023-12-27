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
import HGI

SM = np.loadtxt(r"SM相似性矩阵-weight.txt",dtype=float)   #融合GIPK相似性后的最终SM相似性
MM = np.loadtxt(r"miRNA相似性矩阵-weight.txt",dtype=float) #融合GIPK相似性后的最终miRNA相似性
Y1= np.loadtxt(r"SM-miRNA关联矩阵.txt",dtype=float)

def TCMF(alpha, beta,gamma, Y, maxiter,A,B,C,SM,MM):

    iter0=1
    while True:
        # 范围约束
        A[A < 0] = 0
        A[A > 1] = 1
        B[B < 0] = 0
        B[B > 1] = 1

        #在A矩阵增加L2,1范数约束
        diag_list0 = []

        for i in range(0, 541):
            x0 = A[i, :]
            l = 2 * np.linalg.norm(x0)
            if (l!=0):
                y0 = 1 / l
            else:
                y0 = 0
            diag_list0.append(y0)
        Msm = np.diag(diag_list0)


        # 在B矩阵增加L2,1范数约束
        diag_list = []
        for i in range(0, 541):
            x = B[i, :]
            l = 2 * np.linalg.norm(x)
            if (l != 0):
                y = 1 / l
            else:
                y = 0
            diag_list.append(y)
        Mm = np.diag(diag_list)
        #更新A等式分成如下两部分
        a = np.dot(Y,B)+beta*np.dot(SM,A)
        b = np.dot(np.transpose(B),B)+alpha*C+beta*np.dot(np.transpose(A),A)+ alpha * np.dot(Msm,C)
        #更新B等式分成如下两部分
        c = np.dot(np.transpose(Y),A)+gamma*np.dot(MM,B)
        d = np.dot(np.transpose(A), A) + alpha * C + gamma * np.dot(np.transpose(B), B)+ alpha * np.dot(Mm,C)


        A = np.dot(a,np.linalg.inv(b))
        B = np.dot(c, np.linalg.inv(d))

        if iter0 >= maxiter:

            break
        iter0 = iter0 + 1
    # 范围约束
    A[A < 0] = 0
    A[A > 1] = 1
    B[B < 0] = 0
    B[B > 1] = 1
    Y= np.dot(A,np.transpose(B))
    Y_recover = Y
    return Y_recover


def run_MC_2(Y):
    maxiter = 1000
    alpha = 2
    beta = 3
    gamma = 1
    #SVD

    U, S, V = np.linalg.svd(Y)
    S=np.sqrt(S)
    r = 541
    Wt = np.zeros([r,r])
    for i in range(0,r):
        Wt[i][i]=S[i]
    U= U[:, 0:r]
    V= V[0:r,:]
    A = np.dot(U,Wt)
    B1 = np.dot(Wt,V)
    B=np.transpose(B1)
    C = np.zeros([r, r])
    for i in range(0, r):
        C[i][i] = 1
    Y = TCMF(alpha, beta,gamma,Y, maxiter,A,B,C,SM,MM)
    Smmi = Y
    return Smmi

if __name__ == "__main__":

    M_0 = run_MC_2(Y1)
    Scores_Y = HGII.run_MC_3(M_0)
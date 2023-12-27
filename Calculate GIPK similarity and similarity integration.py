# -*- codeing = utf-8 -*-
# @Author : 刘体耀
# @File : RCLCMF.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import math
import numpy.matlib
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import copy

SM_miRNA_M = np.loadtxt(r"SM-miRNA association matrix.txt", dtype=int)
SM_miRNA_M_2 = np.loadtxt(r"SM-miRNA association matrix 2.txt", dtype=int)
SM = np.loadtxt(r"SM similarity matrix.txt", dtype=int)
miRNA = np.loadtxt(r"miRNA similarity matrix.txt", dtype=int)
SM_2 = np.loadtxt(r"SM similarity matrix 2.txt", dtype=int)
miRNA_2 = np.loadtxt(r"miRNA similarity matrix 2.txt", dtype=int)

#基于数据集1计算SM高斯轮廓核相似性
def Gaussian_SM(w):
    row=831
    sum=0
    SM1=np.matlib.zeros((row,row))
    for i in range(0,row):
        a=np.linalg.norm(SM_miRNA_M[i,])*np.linalg.norm(SM_miRNA_M[i,])
        sum=sum+a
    ps=row/sum
    for i in range(0,row):
        for j in range(0,row):
            SM1[i,j]=math.exp(-ps*np.linalg.norm(SM_miRNA_M[i,]-SM_miRNA_M[j,])*np.linalg.norm(SM_miRNA_M[i,]-SM_miRNA_M[j,]))
    new_SM = w * SM + (1-w) * SM1
    return new_SM
#基于数据集1计算miRNA高斯轮廓核相似性
def Gaussian_MM(w):
    column=541
    sum=0
    miRNA1=np.matlib.zeros((column,column))
    for i in range(0,column):
        a=np.linalg.norm(SM_miRNA_M[:,i])*np.linalg.norm(SM_miRNA_M[:,i])
        sum=sum+a
    ps=column/sum
    for i in range(0,column):
        for j in range(0,column):
            miRNA1[i,j]=math.exp(-ps*np.linalg.norm(SM_miRNA_M[:,i]-SM_miRNA_M[:,j])*np.linalg.norm(SM_miRNA_M[:,i]-SM_miRNA_M[:,j]))
    new_miRNA = w * miRNA + (1 - w) * miRNA1
    return new_miRNA



#基于数据集2计算SM高斯轮廓核相似性
def Gaussian_SM_2(w):
    row=39
    sum=0
    SM1=np.matlib.zeros((row,row))
    for i in range(0,row):
        a=np.linalg.norm(SM_miRNA_M_2[i,])*np.linalg.norm(SM_miRNA_M_2[i,])
        sum=sum+a
    ps=row/sum
    for i in range(0,row):
        for j in range(0,row):
            SM1[i,j]=math.exp(-ps*np.linalg.norm(SM_miRNA_M_2[i,]-SM_miRNA_M_2[j,])*np.linalg.norm(SM_miRNA_M_2[i,]-SM_miRNA_M_2[j,]))
    new_SM_2 = w * SM_2 + (1 - w) * SM1
    return new_SM_2
#基于数据集2计算miRNA高斯轮廓核相似性
def Gaussian_MM_2(w):
    column=286
    sum=0
    miRNA1=np.matlib.zeros((column,column))
    for i in range(0,column):
        a=np.linalg.norm(SM_miRNA_M_2[:,i])*np.linalg.norm(SM_miRNA_M_2[:,i])
        sum=sum+a
    ps=column/sum
    for i in range(0,column):
        for j in range(0,column):
            miRNA1[i,j]=math.exp(-ps*np.linalg.norm(SM_miRNA_M_2[:,i]-SM_miRNA_M_2[:,j])*np.linalg.norm(SM_miRNA_M_2[:,i]-SM_miRNA_M_2[:,j]))
    new_miRNA_2 = w * miRNA_2 + (1 - w) * miRNA1
    return new_miRNA_2

if __name__ == "__main__":
    SM_GIPK = Gaussian_SM(w)
    miRNA_GIPK = Gaussian_MM(w)
    SM_GIPK_2 = Gaussian_SM_2(w)
    miRNA_GIPK_2 = Gaussian_MM_2(w)

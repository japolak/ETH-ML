# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# General math and plotting modules.
import os
import numpy as np
import pandas as pd
import csv

# Load data
path = os.path.join(os.environ['onedrive'] , 'FS19', '02 Intro to ML','Project')
os.chdir(path)

raw=pd.read_csv('Task1a/train.csv', sep=',',header=0, index_col=0)

# Task:
# Your task is to perform 10-fold cross-validation with ridge regression for each of the lambdas given above
# and report the Root Mean Squared Error (RMSE) averaged over the 10 test folds. 
# Approach: 1. define 10-fold, 2. fit Ridge, 3. report RMSE, 4. avarage over 10 RMSE

# prepare
# resample to use consecutive folds
Y = raw.loc[:,'y']
X = raw.loc[:,'x1':]

# closed form solution

def fold(response,features,fo,no):
    n = int(len(response)/fo)
    start = no*n
    stop = (no+1)*n
    mask = raw.index.isin(range(start,stop))
    return(mask);

def ridge_coef(x,y,la):
    dim = x.shape[1]
    w_hat = np.dot(np.linalg.pinv(np.dot(x.T, x) + np.power(10.,la) * np.eye(dim)), np.dot(x.T, y))
    return(w_hat);
    
def ridge_fit(x,w,la):
    Y_hat = np.matmul(x,w)
    return(Y_hat);    
    
def RSE(y_hat,y):
    RSE = np.sqrt(sum(np.square(y_hat-y))/Y_hat.shape[0])
    return(RSE);

def mean(x):
    mu = sum(x)/len(x)
    return(mu);

folds = 10
reg = [-1,0,1,2,3]
solution = []
for r in reg:
    RMSE = []
    for f in range(0,folds):
        m = fold(Y,X,folds,f)        
        Xtrain = X[~m]
        Ytrain = Y[~m]
        w = ridge_coef(Xtrain,Ytrain,r)
        
        Xtest = X[m]
        Ytest = Y[m]
        Y_hat = ridge_fit(Xtest,w,r)
        
        RMSE.append(RSE(Y_hat,Ytest))
    solution.append(mean(RMSE))


with open("Task1a/sol1a_rand.csv", 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\n')
        writer.writerow(solution)
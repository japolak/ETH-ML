# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:49:43 2019

@author: regli
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# General math and plotting modules.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from sklearn import linear_model

# Load data
path = os.path.join(os.environ['onedrive'] , 'FS19', '02 Intro to ML','Project')
os.chdir(path)

raw=pd.read_csv('Task1b/train.csv', sep=',',header=0, index_col=0)

# Task:
# predict y from x1 to x5, where x1 to x5 can be modified with:
# - x
# - x^2
# - exp(x)
# - cos(x)
# - 1

# prepare
Y = raw.loc[:,'y']
X = raw.loc[:,'x1':]

# quadratic term
X[['q1','q2','q3','q4','q5']] = X.apply(np.square)
# exponential term
X[['exp1','exp2','exp3','exp4','exp5']] = X.loc[:,'x1':'x5'].apply(np.exp)
# cosine
X[['cos1','cos2','cos3','cos4','cos5']] = X.loc[:,'x1':'x5'].apply(np.cos)
# constant 1
X['one'] = 1

# use crossvalidation to fit a ridge regression

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
    RSE = np.sqrt(sum(np.square(y_hat-y))/y_hat.shape[0])
    return(RSE);

def mean(x):
    mu = sum(x)/len(x)
    return(mu);
    
def cv(X,fol=10,fit_i=True):
    folds=fol
    l=5
    for i in reversed(range(-2,4)):
      c = 5**i
      reg = np.linspace(max(0,l-c),l+c, endpoint=False)    
      EVAL = []
      co = []
      for r in reg:
          RMSE = []
          for f in range(0,folds):
              m = fold(Y,X,folds,f)        
              Xtrain = X[~m]
              Ytrain = Y[~m]
              clf = linear_model.Lasso(alpha=r, fit_intercept=fit_i)
              w = clf.fit(Xtrain,Ytrain)
                      
              Xtest = X[m]
              Ytest = Y[m]
              Y_pred = clf.predict(Xtest)
                
              RMSE.append(RSE(Y_pred,Ytest))
          EVAL.append(mean(RMSE))
          l = reg[np.argmin(EVAL)]
      plt.plot(reg, EVAL)
      plt.show()
    
    return(l);

opt = cv(X,20) # return RMSE-min lambda
opt

# 10fold: 0.17248000000000002
# 100fold: 0.17216800000000002

# 20fold, intercept=False: 0.04559200000000001


clf = linear_model.Lasso(alpha=opt, fit_intercept=False)
clf.fit(X,Y)
sol = clf.coef_
Y_hat = clf.predict(X)
RSE(Y_hat,Y)


with open("Task1b/sol1b_lasso_2.csv", 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\n')
        writer.writerow(sol)
        
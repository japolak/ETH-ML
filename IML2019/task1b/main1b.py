# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a main script file for Intro to Machine Learning 2019.
@title: task1b
@author: japolak
"""
#%% Import packages
import os
import numpy as np
import pandas as pd
import warnings
from copy import copy as copy
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

#%% Select directory
# You just need to replace/copy the directory below, where this file 'mainn0.py' is located on your computer.
path_dir = os.path.dirname(os.path.realpath('/Users/Jakub/Documents/ETH/S19/IML/task1b/main.py'))
path_train= os.path.join(path_dir + '/train.csv')

#%% Utils functions
def save_solution(csv_file,solution):
	with open(csv_file, 'w') as csv:
            df = pd.DataFrame(solution)
            df.to_csv(csv,index=False, sep='\n')
            
def read_data(csv_file):
    with open(csv_file, 'r') as csv:
        df = pd.read_csv(csv)
        data = df.drop('Id', axis = 1)
    return data

#%% Read Data
data=read_data(path_train)    
Y = copy(pd.DataFrame(data.loc[:,'y']))
X = copy(pd.DataFrame(data.loc[:,'x1':]))

#%% Feature transofrmations
# quadratic term
X[['q1','q2','q3','q4','q5']] = X.apply(np.square)
# exponential term
X[['exp1','exp2','exp3','exp4','exp5']] = X.loc[:,'x1':'x5'].apply(np.exp)
# cosine
X[['cos1','cos2','cos3','cos4','cos5']] = X.loc[:,'x1':'x5'].apply(np.cos)
# constant 1
X['one'] = 1


#%%
def fold(response,features,fo,no):
    n = int(len(response)/fo)
    start = no*n
    stop = (no+1)*n
    mask = data.index.isin(range(start,stop))
    return(mask);

def ridge_coef(x,y,la):
    dim = x.shape[1]
    w_hat = np.dot(np.linalg.pinv(np.dot(x.T, x) + np.power(10.,la) * np.eye(dim)), np.dot(x.T, y))
    return(w_hat);
    
def ridge_fit(x,w,la):
    Y_hat = np.matmul(x,w)
    return(Y_hat);    
    
def RSE(y_pred,y_true):
    RSE = mean_squared_error(y_true, y_pred)**0.5;
    return(RSE);

def mean(x):
    mu = sum(x)/len(x)
    return(mu);
    
def cv(X,fol=10,fit_i=True):
    folds=fol
    l=5
    for i in reversed(range(-3,2)):
      c = 5**i
      reg = np.linspace(max(0,l-c),l+c, endpoint=False)    
      EVAL = []
      for r in reg:
          RMSE = []
          for f in range(0,folds):
              m = fold(Y,X,folds,f)        
              Xtrain = X[~m]
              Ytrain = Y[~m]
              clf = linear_model.Lasso(alpha=r, fit_intercept=fit_i)
              clf.fit(Xtrain,Ytrain)
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

#%%
method = linear_model.ElasticNet(alpha=0.2968,fit_intercept=False, l1_ratio=0.1)
method.fit(X,Y)
weights = method.coef_
Y_hat = method.predict(X)

#%% Performance Measure
y_true = np.array(Y)
y_pred = np.array(Y_hat)
RMSE = mean_squared_error(y_true, y_pred)**0.5;
print('train data RMSE',RMSE)

#%% Save solution
solution = np.array(weights)
path_solution = os.path.join(path_dir + '/solution.csv')
save_solution(path_solution, solution )
# eg.
#dummy_solution = 0.1*np.ones(len(test))
#save_solution(solution_path, dummy_solution)

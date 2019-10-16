# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

@title: task1a
@author: japolak
"""
# General math and plotting modules.
import os
import numpy as np
import pandas as pd
import warnings
from copy import copy as copy
from sklearn.metrics import mean_squared_error

warnings.simplefilter("ignore")

#%% Select directory
path_dir = os.path.dirname(os.path.realpath('/Users/Jakub/Documents/ETH/S19/IML/task1a/main.py'))
path_train= os.path.join(path_dir + '/train.csv')

#%% Utils Functions
def read_data(csv_file):
    with open(csv_file, 'r') as csv:
        df = pd.read_csv(csv)
        data = df.drop('Id', axis = 1)
    return data

def save_solution(csv_file,solution):
	with open(csv_file, 'w') as csv:
            df = pd.DataFrame(solution)
            df.to_csv(csv,index=False, sep='\n', header=False)
  
def RMSE(y_pred,y_true):
    RMSE = mean_squared_error(y_true, y_pred)**0.5;
    return(RMSE);
          
def kfold(Y,folds,it):
    n = int(len(Y)/folds)
    range_start = it*n ; range_stop = (it+1)*n
    index = data.index.isin(range(range_start,range_stop))
    return(index);

def ridge_coef(x,y,la): 
    dim = x.shape[1]
    weights_hat = np.dot(np.linalg.pinv(np.dot(x.T, x) + np.power(10.,la) * np.eye(dim)), np.dot(x.T, y))
    return(weights_hat);
    
def ridge_fit(x,w):
    Y_hat = np.matmul(x,w)
    return(Y_hat);    

#%% Read Data
data=read_data(path_train)
Y = copy(pd.DataFrame(data.loc[:,'y']))
X = copy(pd.DataFrame(data.loc[:,'x1':]))

#%% Compute 
folds = 10
lamb = [-1,0,1,2,3]
sol = []
for j in lamb:
    score = []
    for i in range(0,folds):
        index = kfold(Y,folds,i)        
        Xtrain = X[~index];Xtest = X[index]
        Ytrain = Y[~index];Ytest = Y[index]
        
        weights = ridge_coef(Xtrain,Ytrain,j)
        Y_hat = ridge_fit(Xtest.values,weights)
        
        score.append(RMSE(Y_hat,Ytest))
    sol.append(np.mean(score))

#%% Save Data
solution = np.array(sol)
path_solution = os.path.join(path_dir + '/solution.csv')
save_solution(path_solution, solution )

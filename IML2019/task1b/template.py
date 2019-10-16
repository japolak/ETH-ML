# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a main script file for Intro to Machine Learning 2019.
@title: task template
@author: japolak
"""
#%% Import packages
import os
import numpy as np
import pandas as pd
import warnings
from copy import copy as copy
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")

#%% Select directory
# You just need to replace/copy the directory below, where this file 'mainn0.py' is located on your computer.
path_dir = os.path.dirname(os.path.realpath('/Users/Jakub/Documents/ETH/S19/IML/task1b/main.py'))
path_train= os.path.join(path_dir + '/train.csv')

#%% Utils Functions
def RMSE(y_pred,y_true):
    RMSE = mean_squared_error(y_true, y_pred)**0.5;
    return(RMSE);

def read_data(csv_file):
    with open(csv_file, 'r') as csv:
        df = pd.read_csv(csv)
        data = df.drop('Id', axis = 1)
    return data

def save_solution(csv_file,solution):
	with open(csv_file, 'w') as csv:
            df = pd.DataFrame(solution)
            df.to_csv(csv,index=False, sep='\n')
        
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

#%% Creation of Validation set
dataX = copy(X)
dataY = copy(Y)
X,X_valid,Y,Y_valid = model_selection.train_test_split(X,Y,test_size=0.2,random_state=42)


#%% Cross Validation
'''
folds = 8
kfold = model_selection.KFold(n_splits=folds, shuffle=False, random_state=None)
skfold = model_selection.StratifiedKFold(n_splits=folds, shuffle=False, random_state=None)
kfold.get_n_splits(X)

for train_index, test_index in kfold.split(X): #kfold.split(X,Y):
    print('Train Index: \n', train_index, '\n Test Index: \n', test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
'''
    
#%% Elastic Nets

method = lasso (ridge, lasso, elasticnet)

if method.lower()=='ridge':
    alpha_list = np.linspace(0.01,0.5,num=500)
    model = linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=False, normalize=False, 
                                 scoring=None, cv=5, gcv_mode=None, store_cv_values=False)

l1_list = np.linspace(0.01,1,num=100)
alpha_list = np.linspace(0.001,5,num=100)
model = linear_model.ElasticNetCV(l1_ratio=l1_list, eps=0.001, n_alphas=100, alphas=alpha_list, 
                                  fit_intercept=False, normalize=True, precompute='auto', 
                                  max_iter=1000, tol=0.0001, cv=80, copy_X=True, verbose=0, 
                                  n_jobs=None, positive=False, random_state=None, selection='cyclic')
model.fit(X,Y) 

opt_alpha = model.alpha_ #0.25835
opt_l1 = model.l1_ratio_
weights = model.coef_
Y_valid_hat = pd.DataFrame(model.predict(X_valid))
score = RMSE(Y_valid_hat,Y_valid)
print(' opt alpha:', opt_alpha,'\n opt l1:',opt_l1,'\n score:',score)

#%% Full Model

full_model = linear_model.ElasticNet(l1_ratio=opt_l1,alpha=opt_alpha, fit_intercept=False, normalize=True)
full_model.fit(dataX,dataY)
solution = full_model.coef_

#%% Save Data
path_solution = os.path.join(path_dir + '/solution.csv')
save_solution(path_solution, solution )




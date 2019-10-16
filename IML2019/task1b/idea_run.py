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
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")

#%% Select directory
path_dir = os.path.dirname(os.path.realpath('/Users/Jakub/Documents/ETH/S19/IML/task1b/main.py'))
path_train= os.path.join(path_dir + '/train.csv')

#%% Utils Functions
def read_data(csv_file):
    with open(csv_file, 'r') as csv:
        df = pd.read_csv(csv)
        data = df.drop('Id', axis = 1)
    return data

def RMSE(y_pred,y_true):
    RMSE = mean_squared_error(y_true, y_pred)**0.5;
    return(RMSE);

def save_solution(csv_file,solution):
	with open(csv_file, 'w') as csv:
            df = pd.DataFrame(solution)
            df.to_csv(csv,index=False, sep='\n',header=False)

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
valid_size = 0 # 0 no validation set (ratio)
X,X_valid,Y,Y_valid = model_selection.train_test_split(X,Y,test_size=valid_size,random_state=1)
if valid_size ==0 : del X_valid,Y_valid,valid_size


#%% Cross Validation
n_lambdas=31  #32
n_shuffles=100 #100
n_folds = 100 #20
lambdas=np.linspace(.2,.5,num=n_lambdas)
score_lambdas=np.zeros(n_lambdas)
k=0
for alpha in lambdas: #Go over all possible lambdas
    score_cv=pd.DataFrame(np.zeros(shape=(n_shuffles,n_folds)))
    kfold = model_selection.RepeatedKFold(n_splits=n_folds, n_repeats=n_shuffles, random_state=42)
    i=0; j=0
    for train_index, test_index in kfold.split(X,Y): #Cross Validation for evaluation lambda
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        model = linear_model.Lasso(alpha=alpha, fit_intercept=False, normalize=True,
                                   precompute=False, copy_X=True, max_iter=1000,
                                   tol=0.0001, warm_start=False, positive=False,
                                   random_state=None, selection='cyclic')

        model.fit(X_train,Y_train)
        Y_test_pred = model.predict(X_test)
        score_cv.iloc[j,i]=RMSE(Y_test_pred,Y_test)
        i+=1
        if i==n_folds:
            j+=1;i=0
    score_lambdas[k]=np.mean(np.mean(score_cv,axis=0))
    k+=1
    print(k,'...', end=' ')
del n_folds,n_shuffles,n_lambdas,i,j,k,
opt_lambda = lambdas[np.argmin(score_lambdas)]
print('\n Optimal lambda:', opt_lambda)
plt.plot(lambdas,score_lambdas)

plt.plot(lambdas,score_lambdas)
plt.show(block=True)

#%% Validation Set
if valid_size > 0 :
    model=linear_model.Lasso(alpha=.425,fit_intercept=False, normalize=True)
    model.fit(X,Y)
    Y_valid_hat = pd.DataFrame(model.predict(X_valid))
    score = RMSE(Y_valid_hat,Y_valid)
    print('Optimal lambda:', opt_lambda,'\n Valid score:', score)
j
#%% Full Model
full_model = linear_model.Lasso(alpha=opt_lambda, fit_intercept=False, normalize=True)
full_model.fit(dataX,dataY)
solution = full_model.coef_

#%% Save Data
path_solution = os.path.join(path_dir + '/solution.csv')
save_solution(path_solution, solution )

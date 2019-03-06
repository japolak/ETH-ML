# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a main script file for Intro to Machine Learning 2019.
@title: task0
@author: japolak
"""
#%% Import packages
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

#%% Select directory
# You just need to replace/copy the directory below, where this file 'main0.py' is located on your computer.
path_dir = os.path.dirname(os.path.realpath('/Users/Jakub/Documents/ETH/S19/IML/task0/main0.py'))
path_train= os.path.join(path_dir + '/train.csv')
path_test = os.path.join(path_dir + '/test.csv')

#%% Utils functions
def save_solution(csv_file,pred_prob):
	with open(csv_file, 'w') as csv:
		df = pd.DataFrame.from_dict({'Id':range(10000,10000+len(pred_prob)),'y': pred_prob})
		df.to_csv(csv,index = False)

def load_all_data(csv_file):
    with open(csv_file,'r') as csv:
        df = pd.read_csv(csv)
    return df

def get_data(csv_file):
    with open(csv_file, 'r') as csv:
        df = pd.read_csv(csv)
        df = df.loc[:, df.columns != 'y']
        data = df.drop('Id', axis = 1)
    return data

def get_target(csv_file):
	with open(csv_file, 'r') as csv:
		df = pd.read_csv(csv)
		y = df['y']
	y = np.array(y)
	return y

#%% Read Data
df=load_all_data(path_train)
train=get_data(path_train)
test=get_data(path_test)
y=pd.DataFrame(get_target(path_train),columns = list('y'))

#%% Make Predictions
train_hat=np.array(train.mean(1))
test_hat=np.array(test.mean(1))

#%% Performance Measure
y_true = y
y_pred = train_hat
RMSE = mean_squared_error(y_true, y_pred)**0.5
print('train RMSE',RMSE)

#%% Save solution
solution = test_hat
#dummy_solution = 0.1*np.ones(len(test))
path_solution = os.path.join(path_dir + '/task0.csv')
save_solution(path_solution,solution)

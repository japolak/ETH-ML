#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 07:46:45 2019

@author: leandereberhard
"""
import pickle

def open_pickle(file_name):
    pickle_in = open(file_name, "rb")
    file_out = pickle.load(pickle_in)
    pickle_in.close()
    return(file_out)
    

import os
import itertools
import matplotlib.pyplot as plt
from math import floor, pow
from random import seed as set_seed
from statistics import mean
from numpy.random import choice
import numpy as np
import pandas as pd
from missingpy import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing



#%% First models first try
os.chdir("/Users/leandereberhard/Desktop/ETH/AML/task1/nn_cluster")


# save the lowest MSE from each model 
best_results = {}

for file_name in os.listdir():
   
   results = open_pickle(file_name)
   
   best_results[file_name] = min(results["val_mean_squared_error"])

best_results

# Best test
best_mse = min(best_results.values()) # 51.2279

list(best_results.keys())[list(best_results.values()).index(min(best_results.values()))]





#%% get an estimate of the R^2 of this model 

# import y values used for training
os.chdir("/Users/leandereberhard/Desktop/ETH/AML/task1")
y_train = open_pickle("y_train_cleaned")


# TSS of all y values
((y_train - y_train.sum() / len(y_train)) ** 2).sum()

# don't know which 20% of the observations were sampled, so estimate it 

N = floor(0.2 * len(y_train))

# sample N points from y_train many times
TSS_list = [None] * 10000

for i in range(10000):
    rand_sample = choice(list(y_train['y']), N, replace = False)
    
    # find the TSS of the random sample
    TSS_list[i] = ((rand_sample - mean(rand_sample)) ** 2).sum()
    
    
    
TSS_est = mean(TSS_list)
TSS_est 

RSS = best_mse * N

r_2 = 1 - RSS / TSS_est
r_2 # 0.46777080043184693, not that good 




#%% train another NN using the best parameters 

path = '/Users/leandereberhard/Desktop/ETH/AML/task1/'
os.chdir(path)

x_train_raw = pd.read_csv("X_train.csv")
y_raw = pd.read_csv("y_train.csv")

x_test_raw = pd.read_csv("X_test.csv")

# drop id column before imputing
x_train_raw = x_train_raw.drop("id", axis = 1)
x_test_raw = x_test_raw.drop("id", axis = 1)
y_raw = y_raw.drop("id", axis = 1)



imputer = KNNImputer()

# impute
x_train_imp = pd.DataFrame(imputer.fit_transform(x_train_raw))
x_test_imp = pd.DataFrame(imputer.fit_transform(x_test_raw))

# reset column names
x_train_imp.columns = ["x" + str(i) for i in range(0,832)] 
x_test_imp.columns = ["x" + str(i) for i in range(0,832)] 

# outlier detection 
iso_for = IsolationForest(n_estimators = 100)

iso_for.fit(x_train_imp, y_raw)

outlier_list = np.array(iso_for.predict(x_train_imp))

sum(outlier_list == 1) # 1090
sum(outlier_list == -1) # 122; looks like -1 is the outlier class

# discard all rows that are classified as outliers
x_train = x_train_imp.iloc[outlier_list == 1]
y_train = y_raw.iloc[outlier_list == 1]


# standardize the x values
scaler = preprocessing.StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train))

input_dim = x_train.shape[1]


# Import packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD


#from tensorflow import ConfigProto, Session
#import tensorflow.keras as keras


# Reducing layer width
def reduce_width(neurons, reduce_rate, power):
    return(max(int(neurons*reduce_rate**power), 5))
    
# best parameters from above
dropout = 0.3
neurons = 500
red_rate = 0.3
constraint = 4
mom = 0.9
learning_rate = 0.01
seed = 1
    
set_seed(seed)

model = Sequential()

model.add(Dense(reduce_width(neurons, red_rate, 0), input_shape=(input_dim,), kernel_constraint=MaxNorm(constraint)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(reduce_width(neurons, red_rate, 1), kernel_constraint=MaxNorm(constraint)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(reduce_width(neurons, red_rate, 2), kernel_constraint=MaxNorm(constraint)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(reduce_width(neurons, red_rate, 3), kernel_constraint=MaxNorm(constraint)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('linear'))

# Compile model
# SGD as the optimizer - slower, but should give better results
opt = SGD(lr=learning_rate, momentum=mom, nesterov = True)


model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

# Allow the learning rate to decay with the epoch    
def step_decay(epoch):
   initial_lrate = learning_rate
   drop = 0.5
   epochs_drop = 50.0
   lrate = initial_lrate * pow(drop,  
           floor((1+epoch)/epochs_drop))
   return lrate

# Adaptive step learning rate
lrate = LearningRateScheduler(step_decay, verbose = 1)

# Early stopping
es = EarlyStopping(monitor = 'mse', mode='min', verbose=1, patience=400)
filepath = path  + "/best_model" + "_" + str(neurons) + "_" + str(dropout) + "_" + str(constraint) + "_" + str(red_rate) + "_" + str(mom) + "_" + str(learning_rate) + "_" + str(seed) + ".h5"
mc = ModelCheckpoint(filepath, monitor = 'val_mse', mode='min', verbose=1, save_best_only=True)

# Fit model
model.fit(x = x_train, y = y_train, epochs=1500, batch_size=32, validation_split=0.2, verbose=1, callbacks=[es, lrate, mc])




#%% fit using trained model

# scale the test data
x_test = pd.DataFrame(scaler.transform(x_test_imp))

y_pred = pd.DataFrame(model.predict(x_test))


# write values to file
os.chdir("/Users/leandereberhard/Desktop/ETH/AML/task1")

out = pd.DataFrame({"id": x_test_imp["id"], "y": y_pred[0]})

out.to_csv("nn.csv", index = False)























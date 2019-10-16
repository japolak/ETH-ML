#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 01:11:23 2018

@author: Jakub
"""
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import f1_score
import random
import time
import csv
import pylab as pl
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
"""


"""
# function to load and prepare the data & impute any missing values
# train data
train_eeg1 = pd.read_csv(BASE_PATH + "train_eeg1.csv", dtype=np.float64, header=0, index_col=0)
train_eeg2 = pd.read_csv(BASE_PATH + "train_eeg2.csv", dtype=np.float64, header=0, index_col=0)
train_emg = pd.read_csv(BASE_PATH + "train_emg.csv", dtype=np.float64, header=0, index_col=0)
# test data
test_eeg1 = pd.read_csv(BASE_PATH + "test_eeg1.csv", dtype=np.float64, header=0, index_col=0)
test_eeg2 = pd.read_csv(BASE_PATH + "test_eeg2.csv", dtype=np.float64, header=0, index_col=0)
test_emg = pd.read_csv(BASE_PATH + "test_emg.csv", dtype=np.float64, header=0, index_col=0)
# label data
train_labels = pd.read_csv(BASE_PATH + "train_labels.csv", dtype=np.float64, header=0, index_col=0)
"""

import numpy as np
import pandas as pd
import pylab as pl
import biosppy.signals.eeg as eeg
import biosppy.signals.emg as emg
BASE_PATH =  "/Users/Jakub/Documents/ETH/AML/task5/python/"  
# Download data
train_eeg1 = pd.read_csv(BASE_PATH + "train_eeg1.csv", dtype=np.float64, header=0, index_col=0)
train_eeg2 = pd.read_csv(BASE_PATH + "train_eeg2.csv", dtype=np.float64, header=0, index_col=0)
train_emg = pd.read_csv(BASE_PATH + "train_emg.csv",  dtype=np.float64, header=0, index_col=0)
n=train_eeg1.shape[1] 
m=train_eeg1.shape[0]
fs = 128                
t = 4
ts = np.linspace(0, t, n, endpoint=False)
inp=np.zeros([341,m],dtype=float)
data=np.zeros([512,m],dtype=float)
for e in range(m):
    data = np.array([train_eeg1[e:e].values.flatten(),train_eeg2[e:e].values.flatten()]).T
    outeeg = eeg.eeg(signal=data, sampling_rate=128, labels=None, show=False)
    a=0
    inp[a:a+31,e]=outeeg[3][:,0]
    inp[a+31:a+62,e]=outeeg[4][:,0]
    inp[a+62:a+93,e]= outeeg[5][:,0]
    inp[a+93:a+124,e]= outeeg[6][:,0]
    inp[a+124:a+155,e]= outeeg[7][:,0]
    a=155
    inp[a:a+31,e]=outeeg[3][:,0]
    inp[a+31:a+62,e]=outeeg[4][:,0]
    inp[a+62:a+93,e]= outeeg[5][:,0]
    inp[a+93:a+124,e]= outeeg[6][:,0]
    inp[a+124:a+155,e]= outeeg[7][:,0]
    a=310
    inp[a:a+31,e]=outeeg[9][:,0]
    print(e)

np.savetxt("features.csv", inp.T, delimiter=",")





e=0               # select which epoch   

dataeeg = np.array([train_eeg1[e:e].values.flatten(),train_eeg2[e:e].values.flatten()]).T
dataemg = np.array([train_emg[e:e].values.flatten()]).T
med=np.median(dataemg)
dataemg[dataemg==0]=med
dataemg=dataemg*(-1)
outeeg = eeg.eeg(signal=dataeeg, sampling_rate=128, labels=None, show=False)
outemg = emg.emg(signal=dataemg, sampling_rate=128, show=False)
"""
#outeeg_ts=outeeg[0]             # ts (array) – Signal time axis reference (seconds).
#outeeg_filtered = outeeg[1]     # filtered (array) – Filtered BVP signal.
#outeeg_features_ts = outeeg[2]  # features_ts (array) – Features time axis reference (seconds).
#outeeg_theta = outeeg[3]        # theta (array) – Average power in the 4 to 8 Hz frequency band; 
#outeeg_alpha_low = outeeg[4]    # alpha_low (array) – Average power in the 8 to 10 Hz frequency band; 
#outeeg_alpha_high = outeeg[5]   # alpha_high (array) – Average power in the 10 to 13 Hz frequency band;
#outeeg_beta = outeeg[6]         # beta (array) – Average power in the 13 to 25 Hz frequency band; 
#outeeg_gamma = outeeg[7]        # gamma (array) – Average power in the 25 to 40 Hz frequency band; 
#outeeg_plf_pairs = outeeg[8]    # plf_pairs (list) – PLF pair indices.
#outeeg_plf = outeeg1[9]          # plf (array) – PLF matrix; each column is a channel pair.
"""
a=0
inp[a:a+31,e]=outeeg[3][:,0]
inp[a+31:a+62,e]=outeeg[4][:,0]
inp[a+62:a+93,e]= outeeg[5][:,0]
inp[a+93:a+124,e]= outeeg[6][:,0]
inp[a+124:a+155,e]= outeeg[7][:,0]
a=155
inp[a:a+31,e]=outeeg[3][:,0]
inp[a+31:a+62,e]=outeeg[4][:,0]
inp[a+62:a+93,e]= outeeg[5][:,0]
inp[a+93:a+124,e]= outeeg[6][:,0]
inp[a+124:a+155,e]= outeeg[7][:,0]
a=310
inp[a:a+31,e]=outeeg[9][:,0]




n = x1data.shape[1]      # number of samples
m = x1data.shape[0]      # number of observations
fs = 128                # sampling frequency  
t = 4                   # duration
ts = np.linspace(0, t, n, endpoint=False) # relative timestamps

epoch=0                 # select which epoch

x1row=x1data[epoch:epoch] # extract row signal
x2row=x2data[epoch:epoch] 
    
pl.plot(ts, dataemg, lw=2)# plot signal
pl.plot(ts, temp, lw=2)



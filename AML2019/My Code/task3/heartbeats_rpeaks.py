# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:04:46 2019

@author: Milan
"""
import pandas as pd 
import biosppy.signals.ecg as bse
import numpy as np

#%% Train


data_train = pd.read_csv('Dropbox/MK/ETH MSc Statistics/Machine Learning/Advanced Machine Learning/exercises/project/task3/X_train.csv')
del data_train['id']

rpeaks_init = []
for i in range(0,data_train.shape[0]):
    rpeaks_init.append(bse.engzee_segmenter(signal=data_train.iloc[i,:].values, sampling_rate=300))
    
rpeaks_corrected = []
for i in range(0,data_train.shape[0]):
    rpeaks_corrected.append(bse.correct_rpeaks(signal=data_train.iloc[i,:].values, rpeaks=rpeaks_init[i][0], sampling_rate=300, tol=0.05))

heartbeats = []
for i in range(0,data_train.shape[0]):
    heartbeats.append(bse.extract_heartbeats(signal=data_train.iloc[i,:].values, rpeaks=rpeaks_corrected[i][0], sampling_rate=300, before=0.2, after=0.4)
)
    
R_peaks = []
for i in range(0,data_train.shape[0]):
    R = []
    for j in range(0,len(heartbeats[i][1])):
        R.append(heartbeats[i][1][j]) 
    R_peaks.append(np.array(R))
    
df = pd.DataFrame(R_peaks)
df.to_csv('R_train.csv', index=True, header=False)
    
    
lst = []
for i in range(0,data_train.shape[0]):
    lst.append(heartbeats[i][0])
    
lst2 = []
for i in range(0,data_train.shape[0]):
    lst3 = []
    for j in range(0,len(lst[i])):
        for k in range(0,180):
            lst3.append(lst[i][j][k]) 
    lst2.append(lst3)

my_df = pd.DataFrame(lst2)
my_df.to_csv('HB_train.csv', index=True, header=False)

    
#my_df = pd.DataFrame(index = range(0,data_train.shape[0]), columns = range(0,18000))
#for i in range(0,data_train.shape[0]):
#    for j in range(0,100):
#        my_df.iloc[i,]   
#my_df.to_csv('HB_train.csv', index=True, header=False)


#%% Test


data_test = pd.read_csv('Dropbox/MK/ETH MSc Statistics/Machine Learning/Advanced Machine Learning/exercises/project/task3/X_test.csv')
del data_test['id']

rpeaks_init = []
for i in range(0,data_test.shape[0]):
    rpeaks_init.append(bse.engzee_segmenter(signal=data_test.iloc[i,:].values, sampling_rate=300))
    
rpeaks_corrected = []
for i in range(0,data_test.shape[0]):
    rpeaks_corrected.append(bse.correct_rpeaks(signal=data_test.iloc[i,:].values, rpeaks=rpeaks_init[i][0], sampling_rate=300, tol=0.05))

heartbeats = []
for i in range(0,data_test.shape[0]):
    heartbeats.append(bse.extract_heartbeats(signal=data_test.iloc[i,:].values, rpeaks=rpeaks_init[i][0], sampling_rate=300, before=0.2, after=0.4)
)

R_peaks = []
for i in range(0,data_test.shape[0]):
    R = []
    for j in range(0,len(heartbeats[i][1])):
        R.append(heartbeats[i][1][j]) 
    R_peaks.append(np.array(R))
    
df = pd.DataFrame(R_peaks)
df.to_csv('R_test.csv', index=True, header=False)

lst = []
for i in range(0,data_test.shape[0]):
    lst.append(heartbeats[i][0])
    
lst2 = []
for i in range(0,data_test.shape[0]):
    lst3 = []
    for j in range(0,len(lst[i])):
        for k in range(0,180):
            lst3.append(lst[i][j][k]) 
    lst2.append(lst3)

my_df = pd.DataFrame(lst2)
my_df.to_csv('HB_test.csv', index=True, header=False)


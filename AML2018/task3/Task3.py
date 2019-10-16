#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:27:35 2018

@author: Fede
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg

#### SET WORK FOLDER

# WORK_FOLDER = '/home/leslie/Desktop/AML/Task2/'
WORK_FOLDER = '/Users/Jakub/Documents/ETH/AML/task3/'
#WORK_FOLDER = JAKUB'S


x_train_data=[]
count=0
for line in open(WORK_FOLDER + "X_train.csv"):
    if count==0:                          # remove column names row
        pass
    else:
        # remove id (need to generate nl)
        nl= np.array(line.rstrip('\n').split(",")[1:], dtype=np.float64)
        x_train_data.append(nl)
    count+=1

x_test_data=[]
count=0
for line in open(WORK_FOLDER + "X_test.csv"):
    if count==0:                          # remove column names row
        pass
    else:
        # remove id (need to generate nl)
        nl= np.array(line.rstrip('\n' + '').split(",")[1:], dtype=np.float64)
        x_train_data.append(nl)
    count+=1   

# function to prepare the data & impute any missing values
y_train_data=[]
count=0
for line in open(WORK_FOLDER + "y_train.csv"):
    if count==0:                          # remove column names row
        pass
    else:
        nl=np.array(line.rstrip('\n' + '\r'+ '').split(",")[1:], dtype=np.float64)
        y_train_data.append(nl)
    count+=1
    

def plot_ecg(ts,sd_ts ):
    """Import train and test data. Clean them of index column and 

    Arguments
    ---------
    index: int
      Integer indicating the index of the series we should plot

    Returns
    -------
    A plot of the time series
    """
    # function to prepare the data & impute any missing values
    y=ts
    x=np.arange(len(y))
    plt.plot( x, y, linestyle='solid')
    plt.plot( x, y+ 2*sd_ts,linestyle='solid',color='g')
    plt.plot( x, y- 2*sd_ts,linestyle='solid',color='g')
    plt.show()

def patient_hb_avrg(HB):
    """Get some statistics about the heartbeats

    Arguments
    ---------
    HB: list of heartbeats (from extract_heartbeats)

    Returns
    -------
    average heartbeat statistics
    """
    nr_hb= len(HB)
    nr_ts= len(HB[0])
    avg_ts=np.zeros(shape=nr_ts, dtype=np.float64)
    sd_ts=np.zeros(shape=nr_ts, dtype=np.float64)
    for an_hb in range(nr_hb):
        for a_ts in range(nr_ts):
            avg_ts[a_ts]+=HB[an_hb][a_ts]
    avg_ts=avg_ts/nr_hb
    
    for an_hb in range(nr_hb):
        for a_ts in range(nr_ts):
            sd_ts[a_ts]+=(HB[an_hb][a_ts]-avg_ts[a_ts])**2
    sd_ts=np.sqrt(sd_ts/nr_hb)
    # return the data in a tuple
    return(avg_ts,sd_ts)

"""
############################################### 
# RUN FUNCTIONS
############################################### 
"""
"""
Prepare the data & return in usable format +
Cleaning the data (remove outliers and impute using median)
"""

def avrg_sd_allSubj(segmenter="Hamilton"):
    nr_subj=len(x_train_data)
    subj_avgs=[]
    subj_sd=[]
    for subj in range(nr_subj):
        ts=x_train_data[subj]
        if segmenter=="Hamilton":
            rpeak0=ecg.hamilton_segmenter(signal=ts, sampling_rate=300)[0]
        elif segmenter=="Christov":
            rpeak0=ecg.christov_segmenter(signal=ts, sampling_rate=300)[0]
            # same as 'ecg.engzee_segmente' if we pass raw input
        elif segmenter=="Gamboa":
            rpeak0=ecg.gamboa_segmenter(signal=ts, sampling_rate=300, tol=0.002)[0]
            # gamboa on the contrary delivers (very?!) different results
        hb= ecg.extract_heartbeats(signal=ts, rpeaks= rpeak0, sampling_rate=300.0)[0] 
        subj_avgs.append(patient_hb_avrg(hb)[0])
        subj_sd.append(patient_hb_avrg(hb)[1])
    return subj_avgs, subj_sd

subj_avrg, subj_sd=avrg_sd_allSubj()

"""
APPARENTLY ALL HEARTHBEATS ARE 180 UNITS LONG!
"""
def avrg_sd_xClass(subj_avrgS, subj_sdS, response_ls=0):
    index_cl = [i for i, x in enumerate(y_train_data) if x == response_ls]
    subj_avrgS = [subj_avrgS[i] for i in index_cl]
    subj_sdS = [subj_sdS[i] for i in index_cl]
    class_avgs=[]
    class_sd=[]
    nr_subj=len(subj_avrgS)
    nr_ts= len(subj_avrgS[0])
    avg_ts=np.zeros(shape=nr_ts, dtype=np.float64)
    sd_ts=np.zeros(shape=nr_ts, dtype=np.float64)
    for a_sj in range(nr_subj):
        for a_ts in range(nr_ts):
            avg_ts[a_ts]+=subj_avrgS[a_sj][a_ts]
    class_avgs=avg_ts/nr_subj
    
    for a_sj in range(nr_subj):
        for a_ts in range(nr_ts):
            sd_ts[a_ts]+=(subj_avrgS[a_sj][a_ts]-class_avgs[a_ts])**2
    class_sd=np.sqrt(sd_ts/nr_subj)
    # return the data in a tuple
    return(class_avgs,class_sd)

class_avgs0, class_sd0=avrg_sd_xClass(subj_avrg, subj_sd)
class_avgs1, class_sd1=avrg_sd_xClass(subj_avrg, subj_sd, response_ls=1)
class_avgs2, class_sd2=avrg_sd_xClass(subj_avrg, subj_sd, response_ls=2)
class_avgs3, class_sd3=avrg_sd_xClass(subj_avrg, subj_sd, response_ls=3)


%matplotlib inline
y=subj_avrg[15]
x=np.arange(len(y))
plt.plot( x, y, linestyle='solid')
plt.plot( x, y+ 2*subj_sd[10],linestyle='solid',color='g')
plt.plot( x, y- 2*subj_sd[10],linestyle='solid',color='g')
plt.show()


plot_ecg(class_avgs3, class_sd3)



# Peak correction (tol in seconds--> how should I tune?)
# new_peak=ecg.correct_rpeaks(signal=signal0, rpeaks=Rpeak0,tol=0.3, sampling_rate=300)

"""
Features
"""
features=ecg.ecg(x_train_data[10], sampling_rate=300,show=True)

##############
# Write a function that computes the features of interest
# based on what we get from .ecg or .extract_heartbeats







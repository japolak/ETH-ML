#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 13:05:13 2018

@author: fred
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg

#### SET WORK FOLDER

# WORK_FOLDER = '/home/leslie/Desktop/AML/Task2/'
WORK_FOLDER = '/home/fred/Documents/ETH/HS2018/Advanced_Machine_Learning/Projects/Task3/task3-data/'
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
        x_test_data.append(nl)
    count+=1   

y_train_data=[]
count=0
for line in open(WORK_FOLDER + "y_train.csv"):
    if count==0:                          # remove column names row
        pass
    else:
        nl=np.array(line.rstrip('\n' + '\r'+ '').split(",")[1:], dtype=np.float64)
        y_train_data.append(nl)
    count+=1
"""
#########################
Manual Feature Extraction
#########################
"""
"""
Train data: class frequencies
Class 0: 3030[prior:0.592]
Class 1: 443 [prior:0.087]
Class 2: 1474[prior:0.288]
Class 3: 170 [prior:0.033]
"""

def extract_features(ts):
    """Extract features from individual patient

    Arguments
    ---------
    ts: time series of interest

    Returns
    -------
    
    """
    
    """
    1. R-R interval:
    -what is the average interval between peaks?
    -how much variation is there in this measure?
    -max/min value
    """ 
    # Peak detection: perhaps take it from .ecg function?
    rpeak0=ecg.hamilton_segmenter(signal=ts,\
                              sampling_rate=300)[0]
    # Interval lengths
    l_RRint=[]
    for a in range(len(rpeak0)):
        if(a>0):
            l_RRint.append(rpeak0[a]-rpeak0[a-1])
    # Some "robust" extreme values statistics (no max or min)
    RRint_10, RRint_90=np.percentile(l_RRint, q=[0.1,0.9])
    RRint_mean=np.mean(l_RRint)
    RRint_sd=np.std(l_RRint)
    """
    2. R amplitude (difference between value at peak
                    minus baseline -not "valley"-):
        -what is the average R amplitude?
        -how much variation is there in this measure?
        -max/min value
    NOTE: YOU CANNOT USE THE INDEXING OF THE R-PEAKS TO FIND THE
    VALUE OF THE Y! First impression: use ecg.ecg and work with
    the filtered series (here indeces seem to match)
    
    Idea: work with index of filtered series. For baseline
    take an average of the first and last 20 values of the series
    """          
    filtered_ts=ecg.ecg(ts, sampling_rate=300, show=False)
    
    hb, hb_peaks= ecg.extract_heartbeats(signal=filtered_ts["filtered"], \
                    rpeaks= rpeak0, sampling_rate=300.0)
    # if all(hb[1]==rpeak0):
    #    hb=hb[0]
    #else:
    #    print "Error"
    # Finding value at R-peak
    R_max_value=filtered_ts["filtered"][hb_peaks]
    # Computing baseline
    ts_baselines=[]
    for a_hb in hb:
        ts_baselines.append((sum(a_hb[0:20])+ \
                            sum(a_hb[(len(a_hb)-20):(len(a_hb)-1)]))/40.)
    # R_max -baseline
    R_amplitude=[]
    for hb_nr in range(len(hb)):
        R_amplitude.append(R_max_value[hb_nr]-ts_baselines[hb_nr])
    Rampl_10, Rampl_90=np.percentile(R_amplitude, q=[0.1,0.9])
    Rampl_mean=np.mean(R_amplitude)
    Rampl_sd=np.std(R_amplitude)
    
    """
    3. Q and S (latter optional) amplitude 
               (difference between value at min before R
                    minus baseline):
    
    Idea: Examinate the HB. Find out: what is a reasonal nr of
     steps to look before the R peak to find Q? Baseline (see above)
     - 25 time steps before R-peak to find Q value (the same for S)
     --> Note, since the R-peak is sometimes identified at the very
     beginning of the series, we need the window to be relative
     to where this index is choice: look at indeces
     int(2./3*R_index):R_index
    """
    Q_vals=[]
    Q_amplitude=[]
    S_vals=[]
    S_amplitude=[]
    for hb_nr in range(len(hb)):
        # Q stats
        R_index=np.where(hb[hb_nr] == R_max_value[hb_nr])[0][0]
        Q_min= min(hb[hb_nr][int(2./3*R_index):R_index])
        Q_vals.append(Q_min)
        Q_amplitude.append(ts_baselines[hb_nr]-Q_min)
        # S stats
        S_min= min(hb[hb_nr][R_index:(R_index+25)])
        S_vals.append(S_min)
        S_amplitude.append(ts_baselines[hb_nr]-S_min)
    # Remark: some values are completely off due to clear errors in the 
    # peak detction algorithm (e.g. heartbeat 34 in 19th individual)
    Qampl_10, Qampl_90=np.percentile(Q_amplitude, q=[0.1,0.9])
    Qampl_mean=np.mean(Q_amplitude)
    Qampl_sd=np.std(Q_amplitude)
    Sampl_10, Sampl_90=np.percentile(S_amplitude, q=[0.1,0.9])
    Sampl_mean=np.mean(S_amplitude)
    Sampl_sd=np.std(S_amplitude)
    """
    4. QRS duration:
    
    Idea: we have the index of Q_min and that of S_min. Difference between the two
    APPROXIMATES this feature (the way I understand it, I should measure how
    long it takes, once we have left the "baseline" to fall into the Q-in, to
    come back to the baseline after the S min)
    """
    QRS_time=[]
    for hb_nr in range(len(hb)):
        Q_index=np.where(hb[hb_nr] == Q_vals[hb_nr])[0][0]
        S_index=np.where(hb[hb_nr] == S_vals[hb_nr])[0][0]
        QRS_time.append(S_index-Q_index)
    QRSts_10, QRSts_90=np.percentile(QRS_time, q=[0.1,0.9])
    QRSts_mean=np.mean(QRS_time)
    QRSts_sd=np.std(QRS_time)    
    """
    5. Hearth rate variability
    
    Idea: extract from the .ecg function
    """
    hr_ts=filtered_ts["heart_rate"] # For subjet 2719 no heart rate is computed! Weird... 
    if len(hr_ts)==0 :
        HR_10, HR_90, HR_mean, HR_sd= [None]*4
    else:
        HR_10, HR_90=np.percentile(hr_ts, q=[0.1,0.9])
        HR_mean=np.mean(hr_ts)
        HR_sd=np.std(hr_ts)
    """
    5. Wavelet energy
    
    Idea: don't understand what this is or how to obtain it
    """
    
    return RRint_mean, RRint_sd, RRint_10, RRint_90, \
           Rampl_mean, Rampl_sd, Rampl_10, Rampl_90, \
           Qampl_mean, Qampl_sd, Qampl_10, Qampl_90, \
           Sampl_mean, Sampl_sd, Sampl_10, Sampl_90, \
           QRSts_mean, QRSts_sd, QRSts_10, QRSts_90, \
           HR_mean, HR_sd, HR_10, HR_90

df_features=[]
for a_subj in range(len(x_train_data)):
    print a_subj
    df_features.append(np.array(extract_features(x_train_data[a_subj]), dtype=np.float64))

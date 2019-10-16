#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:25:47 2018

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
    

def patient_hb_avrg(hb):
    """Get some statistics about the heartbeats

    Arguments
    ---------
    hb: list of heartbeats (from extract_heartbeats)

    Returns
    -------
    average heartbeat statistics
    """
    nr_hb= len(hb)
    nr_ts= len(hb[0])
    avg_ts=np.zeros(shape=nr_ts, dtype=np.float64)
    sd_ts=np.zeros(shape=nr_ts, dtype=np.float64)
    for an_hb in range(nr_hb):
        for a_ts in range(nr_ts):
            avg_ts[a_ts]+=hb[an_hb][a_ts]
    avg_ts=avg_ts/nr_hb
    
    for an_hb in range(nr_hb):
        for a_ts in range(nr_ts):
            sd_ts[a_ts]+=(hb[an_hb][a_ts]-avg_ts[a_ts])**2
    sd_ts=np.sqrt(sd_ts/nr_hb)
    # return the data in a tuple
    return(avg_ts,sd_ts)

def avrg_sd_allSubj(x_data, segmenter="Hamilton", save_as=None):
    """For all subjects, compute the average heartbeat
       (calling patient_hb_avrg)
    
    Arguments
    ---------
    x_data: raw time series
    segmenter: Hamilton, Christov or Gamboa
    save_as: save the average -TEST or TRAIN data- 

    Returns
    -------
    if save_as= None, returns two lists [avrg heartbeat and its sd]
    otherwise saves the average heartbeats for all subjs
    """
    nr_subj=len(x_data)
    subj_avgs=[]
    subj_sd=[]
    for subj in range(nr_subj):
        print subj
        ts=x_data[subj]
        # Better to filter everything out!
        ts=ecg.ecg(signal=ts, sampling_rate=300, show=False)["filtered"]
        if segmenter=="Hamilton":
            rpeak0=ecg.hamilton_segmenter(signal=ts, sampling_rate=300)[0]
            """
             H0 a: Hamilton is the default on the filtered series of ecg.ecg
             H0 b: Hamilton on the raw series returns another output
             --> H0 b is verified, still, as hinted in the file
                 we should work with the filtered anyway...
            """
        elif segmenter=="Engzee":
            rpeak0=ecg.engzee_segmenter(signal=ts, sampling_rate=300, threshold=0.1)[0]
            """
            We obtain an error if we do not specify the threshold
            for the 14th obs in test data.
            With the argument specified at 0.1 (gridsearch over
            values was carried out for 3 data points only), we solve the
            issue! [note: a low treshold favours the discovery
            of a LOT of peaks -around the same amount than with normal
            ecg.ecg-. This is what we have a preference for.]
            --> peaks differ from Hamilton!
            """
        elif segmenter=="Ssf":
            rpeak0=ecg.ssf_segmenter(signal=ts, sampling_rate=300)[0]
            """
             a lot of arguments which I do not know how to tune.
             STILL: no error whatsoever
             AND: the resulting HB are VERY VERY DIFFERENT!
            """
        elif segmenter=="Christov":
            rpeak0=ecg.christov_segmenter(signal=ts, sampling_rate=300)[0]
            """
             issue with individual 39 test data (a.o.) --> merely one
             peak detected (even when passing .ecg.ecg filtered output!)
             The original TS is long though an extract_heartbeats
             returns an error
            """
        elif segmenter=="Gamboa":         
            """
             issue with individual 331 test data (a.o.) --> ERROR
             (even if we change the argument tol)
             BUT: if we filter it, then it return peaks, which are
             different from those of ecg.ecg()
             STILL, for sample 500 of test data, we still get
             a weird error!
            """
            rpeak0=ecg.gamboa_segmenter(signal=ts, sampling_rate=300, tol=0.002)[0]
        hb, hb_peaks= ecg.extract_heartbeats(signal=ts, rpeaks= rpeak0, sampling_rate=300.0)
        subj_mean0,subj_sd0= patient_hb_avrg(hb)
        subj_avgs.append(subj_mean0)
        subj_sd.append(subj_sd0)
    if not(save_as is None):
        x_dat = pd.DataFrame(data=subj_avgs)      
        x_dat.to_csv(("%s%s_hb_subjAvg_%s.csv") % (WORK_FOLDER,segmenter,save_as), index=False) 
    else:
        return subj_avgs, subj_sd

%matplotlib inline

avrg_sd_allSubj(x_data=x_test_data, segmenter="Hamilton", save_as="TEST")
avrg_sd_allSubj(x_data=x_test_data, segmenter="Engzee", save_as="TEST")
avrg_sd_allSubj(x_data=x_test_data, segmenter="Ssf", save_as="TEST")


avrg_sd_allSubj(x_data=x_train_data, segmenter="Hamilton", save_as="TRAIN")
avrg_sd_allSubj(x_data=x_train_data, segmenter="Engzee", save_as="TRAIN")
avrg_sd_allSubj(x_data=x_train_data, segmenter="Ssf", save_as="TRAIN")
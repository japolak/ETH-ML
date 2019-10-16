#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:23:35 2018

@author: fred
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
import time
import sys
import os
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
WORK_FOLDER = '/home/fred/Documents/ETH/HS2018/Advanced_Machine_Learning/Projects/Task5/task5-data/'
LOG_PATH = os.path.join(WORK_FOLDER, "Logs/mymodel_final.ckpt")
#WORK_FOLDER = JAKUB'S
def impute_median(data_to_clean):
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)    
    data_to_clean= imp.fit_transform(data_to_clean)
    data_to_clean = pd.DataFrame(data=data_to_clean)
    return data_to_clean

# load data
x_train_eeg1 = pd.read_csv('%sfeatures/train_eeg1_features.csv'  % (WORK_FOLDER) , header=None, dtype=np.float32)
x_train_eeg2 = pd.read_csv('%sfeatures/train_eeg2_features.csv'  % (WORK_FOLDER) , header=None,dtype=np.float32)
x_train_emg =  pd.read_csv('%sfeatures/train_emg_features.csv'   % (WORK_FOLDER) ,header=None, dtype=np.float32)
y_train_data = pd.read_csv('%strain_labels.csv'% (WORK_FOLDER) , dtype=np.float64)
x_test_eeg1 = pd.read_csv('%sfeatures/test_eeg1_features.csv'  % (WORK_FOLDER) , header=None, dtype=np.float32)
x_test_eeg2 = pd.read_csv('%sfeatures/test_eeg2_features.csv'  % (WORK_FOLDER) , header=None,dtype=np.float32)
x_test_emg =  pd.read_csv('%sfeatures/test_emg_features.csv'   % (WORK_FOLDER) ,header=None, dtype=np.float32)    
x_train = pd.concat([x_train_eeg1, x_train_eeg2,x_train_emg ], axis=1, sort=False)
x_test = pd.concat([x_test_eeg1, x_test_eeg2,x_test_emg ], axis=1, sort=False)
y_train_data = y_train_data.loc[:,'y']
x_train = x_train.replace(-np.inf, np.nan) # NAs instead than -inf
x_train = impute_median(x_train) # removing NAs
x_train= x_train.round(5)
x_test= x_test.replace(-np.inf, np.nan) # NAs instead than -inf
x_test = impute_median(x_test) # removing NAs
x_test = x_test.round(5)  

tuned_parameters = [{'kernel': ['rbf','linear', 'poly', 'sigmoid'], 'gamma': [0.01, 0.02, 0.03, 0.04, 0.05],
                     'C': [1], "decision_function_shape": ["ovo"]}]
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1],'C': [1], "decision_function_shape": ["ovr"]}]
                    # {'kernel': ['linear'], 'C': [0.0001, 0.001, 0.01, 0.1, 1], "decision_function_shape": ["ovo", "ovr"]},
                    # {'kernel': ['poly'], 'degree': [2, 3, 4, 5], "decision_function_shape": ["ovo", "ovr"]}]
clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring="balanced_accuracy", verbose=3)
clf.fit(x_train, y_train_data)
print("best parameters")
print(clf.best_params_)

svm_model = SVC(kernel = 'rbf', gamma=0.001, C = 1, probability=True, decision_function_shape="ovo", class_weight=class_weight).fit(x_train_data, y_train_data) 
svm_prob = svm_model.predict_proba(x_test)

# multinormial
CV=10
nr_obs=x_train.shape[0]
batch_size=nr_obs/CV
shuffled_index=shuffle(range(nr_obs)) # for each epoch we should pass a different batch at any time.
for batch_number in range(CV):
    starting_point=batch_number*batch_size
    ending_point=(batch_number+1)*batch_size
    btch=shuffled_index[starting_point:ending_point]
    x_train_CV=x_train.drop(btch)
    y_train_CV=y_train_data.drop(btch)
    x_test_CV=x_train.iloc[btch,:]
    y_test_CV=y_train_data[btch]
    log_model = LogisticRegression(C=0.001, multi_class='multinomial',
                                penalty="l2", class_weight={1:1, 2:1.1, 3:18},
                                max_iter=1000, solver='sag').fit(x_train_CV, y_train_CV)
    log_y= log_model.predict(x_test_CV)
    BMAC = balanced_accuracy_score(y_test_CV, log_y)
    print("CV%s logit" % (batch_number))
    print BMAC
    lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=.9, priors=[.5, .47, 0.03]).fit(x_train, y_train_data)
    lda_y = lda_model.predict(x_test)
    BMAC = balanced_accuracy_score(y_test_CV, lda_y)
    print("CV%s LDA" % (batch_number))
    print BMAC
    
# This does not work!
# log_model = LogisticRegressionCV(cv=10, multi_class="ovr", solver="lbfgs", max_iter=1000,
#                                 class_weight={1:1, 2:1.1, 3:18}).fit(x_train, y_train_data)
    
# Fitting the logits specified above
log_model = LogisticRegression(C=0.001, multi_class='multinomial',
                                penalty="l2", class_weight={1:1, 2:1.1, 3:18},
                                max_iter=1000, solver='sag').fit(x_train, y_train_data)
log_y= log_model.predict(x_test)
y_test_pred = pd.DataFrame({"Id": range(len(log_y)), "y": log_y})
## save results to csv
y_test_pred.to_csv(WORK_FOLDER + "task5_logistic.csv", index=False)	


# Fitting tLDA specified above
lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=.9, priors=[.5, .47, 0.03]).fit(x_train, y_train_data)
lda_y = lda_model.predict(x_test)
y_test_pred = pd.DataFrame({"Id": range(len(lda_y)), "y": lda_y})
## save results to csv
y_test_pred.to_csv(WORK_FOLDER + "task2_LDA.csv", index=False)	
# np.unique(log_y ,return_counts=True)
	
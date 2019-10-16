#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:27:35 2018
@author: leslie
"""
import numpy as np
import pandas as pd
import os
import time
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# from sklearn.metrics import balanced_accuracy_score
"""
def balanced_accuracy_score(y_true, y_pred, sample_weight=None,adjusted=False):
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score
"""
#### SET WORK FOLDER
# WORK_FOLDER = '/home/fred/Documents/ETH/HS2018/Advanced Machine Learning/Projects/Task2/task2-data/'
# WORK_FOLDER = FEDE
WORK_FOLDER = '/Users/Jakub/Documents/ETH/AML/task2/'

def prepare_data():
    # function to prepare the data & impute any missing values
    # load data
    x_train = pd.read_csv(WORK_FOLDER + "X_train.csv", dtype=np.float64)
    y_train = pd.read_csv(WORK_FOLDER + "y_train.csv", dtype=np.float64)
    x_test = pd.read_csv(WORK_FOLDER + "X_test.csv", dtype=np.float64)
    x_trn = pd.read_csv(WORK_FOLDER + "X_train_a.csv", dtype=np.float64)
    x_val = pd.read_csv(WORK_FOLDER + "X_train_b.csv", dtype=np.float64)
    y_trn = pd.read_csv(WORK_FOLDER + "y_train_a.csv", dtype=np.float64)
    y_val = pd.read_csv(WORK_FOLDER + "y_train_b.csv", dtype=np.float64)
    # Remove ID column from x train
    col_names_x = x_train.columns.get_values()
    col_names_x = col_names_x.tolist()
    features = col_names_x[1:]
    x_train = x_train[features]
    x_trn = x_trn[features]
    x_val = x_val[features]
    # Remove id column from y train
    col_names_y = y_train.columns.get_values()
    col_names_y = col_names_y.tolist()
    response = col_names_y[1:]
    y_train = y_train[response]
    y_trn = y_trn[response]
    y_val = y_val[response]
    # Remove id from x test and get ID column (to make the CSV later)
    ids = col_names_x[0]
    id_data = x_test[ids]
    x_test = x_test[features]
    ## Scale data
    scaler = StandardScaler()
    scaler.fit(x_train)
    scaler.fit(x_trn)
    scaler.fit(x_val)
    x_train = scaler.transform(x_train)
    x_trn = scaler.transform(x_trn)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    # convert to pandas dataframe
    x_train = pd.DataFrame(data=x_train)
    y_train = pd.DataFrame(data=y_train)
    x_test = pd.DataFrame(data=x_test)
    x_trn = pd.DataFrame(data=x_trn)
    x_val = pd.DataFrame(data=x_val)
    y_trn = pd.DataFrame(data=y_trn)
    y_val = pd.DataFrame(data=y_val)
    # return the data in a tuple
    return(x_train, y_train, x_test, x_trn, x_val, y_trn, y_val, id_data)

x_train, y_train, x_test, x_trn, x_val, y_trn, y_val, id_data= prepare_data()
# y = column_or_1d(y,warn=True)

"""
FULL MODEL
"""
# Support Vector Machine
# svm_model = SVC(kernel = 'rbf', gamma=0.001, C = 1, probability=True, decision_function_shape="ovo", class_weight={0:6, 1:1, 2:6}).fit(x_train_data, y_train_data)
# Multinomial Logistic Regression
# log_model = LogisticRegression(C=0.001, multi_class='multinomial', penalty="l2", solver='sag', max_iter=1000, class_weight={0:6, 1:1, 2:6}).fit(x_train_data, y_train_data)
# Linear Discriminant Analysis
# lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=.9, priors=[1./8, 3./4, 1./8]).fit(x_train_data, y_train_data)

"""
BLENDING

1. The initial train set is split into training (trn) and validation (val) sets.
2. Model(s) are fitted on the training (trn) set.
3. The predictions are made on the validation (val) set and the test set.
4. The validation set and its predictions are used as features to build a new model.
5. This model is used to make final predictions on the test and meta-features.

"""
# Use the previous models (SVM,LOG,LDA) on the train set in order
# to make predictions on the validation set.

model_svm = SVC(kernel = 'rbf', gamma=0.001, C = 1, probability=True, decision_function_shape="ovo", class_weight={0:6, 1:1, 2:6})
model_svm.fit(x_trn, y_trn)
x_val_pred_svm=model_svm.predict(x_val)
x_test_pred_svm=model_svm.predict(x_test)
x_val_pred_svm=pd.DataFrame(x_val_pred_svm)
x_test_pred_svm=pd.DataFrame(x_test_pred_svm)

model_log = LogisticRegression(C=0.001, multi_class='multinomial', penalty="l2", solver='sag', max_iter=1000, class_weight={0:6, 1:1, 2:6})
model_log.fit(x_trn,y_trn)
x_val_pred_log=model_svm.predict(x_val)
x_test_pred_log=model_svm.predict(x_test)
x_val_pred_log=pd.DataFrame(x_val_pred_log)
x_test_pred_log=pd.DataFrame(x_test_pred_log)

model_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=.9, priors=[1./8, 3./4, 1./8])
model_lda.fit(x_trn, y_trn)
x_val_pred_lda=model_svm.predict(x_val)
x_test_pred_lda=model_svm.predict(x_test)
x_val_pred_lda=pd.DataFrame(x_val_pred_lda)
x_test_pred_lda=pd.DataFrame(x_test_pred_lda)


# Combining the meta-features and the validation set, a logistic regression
# model is built to make predictions on the test set.

df_val=pd.concat([x_val, x_val_pred_svm,x_val_pred_log,x_val_pred_lda],axis=1)
df_test=pd.concat([x_test, x_test_pred_svm,x_test_pred_log,x_test_pred_lda],axis=1)

model_ens = LogisticRegression(C=0.001, multi_class='multinomial', penalty="l2", solver='sag', max_iter=1000, class_weight={0:6, 1:1, 2:6})
model_ens.fit(df_val,y_val)
y_test_pred_ens=model_ens.predict(df_test)

y_test_pred_ens = pd.DataFrame(data=y_test_pred_ens)
y_test_pred_ens.columns = ["y"]
y_predictions = y_test_pred_ens["y"]
y_test_pred_ens = pd.DataFrame({"id": id_data, "y": y_predictions})
y_test_pred_ens.to_csv(WORK_FOLDER + "task2_ens.csv", index=False)


"""
STACKING

1. The initial train set is split into training (trn) and validation (val) sets.
2. Model(s) are fitted on the training (trn) set.
3. The predictions are made on the validation (val) set and the test set.
4. The validation set and its predictions are used as features to build a new model.
5. This model is used to make final predictions on the test and meta-features.

"""

def Stacking(model,xtrain,ytrain,xtest,n_fold=10):
    folds=StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred=np.empty((xtest.shape[0],1),float)
    train_pred=np.empty((0,1),float)
    for train_indices,val_indices in folds.split(xtrain,ytrain.values):
        x_train,x_val=xtrain.iloc[train_indices],xtrain.iloc[val_indices]
        y_train,y_val=ytrain.iloc[train_indices],ytrain.iloc[val_indices]
        model.fit(X=x_train,y=y_train)
        train_pred=np.append(train_pred,model.predict(x_val))
        test_pred=np.append(test_pred,model.predict(xtest))
    return test_pred.reshape(-1,1),train_pred

model_svm = SVC(kernel = 'rbf', gamma=0.001, C = 1, probability=True, decision_function_shape="ovo", class_weight={0:6, 1:1, 2:6})
test_pred_svm ,train_pred_svm=Stacking(model=model_svm,n_fold=10, xtrain=x_train,xtest=x_test,ytrain=y_train)
train_pred_svm=pd.DataFrame(train_pred_svm)
test_pred_svm=pd.DataFrame(test_pred_svm)

model_log = LogisticRegression(C=0.001, multi_class='multinomial', penalty="l2", solver='sag', max_iter=1000, class_weight={0:6, 1:1, 2:6})
test_pred_log ,train_pred_log=Stacking(model=model_log,n_fold=10, xtrain=x_train,xtest=x_test,ytrain=y_train)
train_pred_log=pd.DataFrame(train_pred_log)
test_pred_log=pd.DataFrame(test_pred_log)

model_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=.9, priors=[1./8, 3./4, 1./8])
test_pred_lda ,train_pred_lda=Stacking(model=model_lda,n_fold=10, xtrain=x_train,xtest=x_test,ytrain=y_train)
train_pred_lda=pd.DataFrame(train_pred_lda)
test_pred_lda=pd.DataFrame(test_pred_lda)

df = pd.concat([x_train, train_pred_svm, train_pred_log, train_pred_lda], axis=1)
df_test = pd.concat([x_test, test_pred_svm, test_pred_log, test_pred_lda], axis=1)

model_ens = LogisticRegression(C=0.001, multi_class='multinomial', penalty="l2", solver='sag', max_iter=1000, class_weight={0:6, 1:1, 2:6})
model_ens.fit(df,y_train)
y_test_pred_ens=model_ens.predict(df_test)

y_test_pred_ens = pd.DataFrame(data=y_test_pred_ens)
y_test_pred_ens.columns = ["y"]
y_predictions = y_test_pred_ens["y"]
y_test_pred_ens = pd.DataFrame({"id": id_data, "y": y_predictions})
y_test_pred_ens.to_csv(WORK_FOLDER + "task2_ens.csv", index=False)
